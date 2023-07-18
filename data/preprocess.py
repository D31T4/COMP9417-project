'''
pre-process CMU mocap dataset
'''
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from shutil import rmtree
from typing import Union

def to_meter(p: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
    '''
    convert to meter

    ref: http://mocap.cs.cmu.edu/info.php
    '''
    return p * (1.0 / 0.45) * 2.54 / 100.0

def rotation_matrix(x: float = 0, y: float = 0, z: float = 0) -> np.ndarray:
    '''
    convert euler angles to rotation matrix

    Arguments:
    ---
    - x: rotation around x-axis (rad)
    - y: rotation around y-axis (rad)
    - z: rotation around z-axis (rad)

    Returns:
    ---
    - transformation matrix
    '''
    rx = np.array([
        [1, 0, 0],
        [0, np.cos(x), -np.sin(x)],
        [0, np.sin(x), np.cos(x)]
    ])

    ry = np.array([
        [np.cos(y), 0, np.sin(y)],
        [0, 1, 0],
        [-np.sin(y), 0, np.cos(y)]
    ])

    rz = np.array([
        [np.cos(z), -np.sin(z), 0],
        [np.sin(z), np.cos(z), 0],
        [0, 0, 1]
    ])

    return rx @ ry @ rz

class Joint:
    '''
    represent a joint in the skeleton
    '''
    def __init__(self, name: str, direction: np.ndarray, length: float, axis: np.ndarray, dof: list[str], limits: list[tuple[float, float]]):
        self.name = name
        self.parent: Joint = None
        self.children: list[Joint] = []

        assert direction.shape == (3, 1)
        self.direction = direction
        
        self.length = length
        self.dof = dof
        
        assert axis.shape == (3,)
        axis = np.deg2rad(axis)
        self.C = rotation_matrix(*axis)
        self.Cinv = np.linalg.inv(self.C)

        self.limits = [(0, 0)] * 3

        for ax, lim in zip(dof, limits):
            if ax == 'rx':
                self.limits[0] = lim
            elif ax == 'ry':
                self.limits[1] = lim
            elif ax == 'rz':
                self.limits[2] = lim

        self.matrix: np.ndarray = None
        self.coordinate: np.ndarray = None

    def step(self, motion: np.ndarray):
        euler_angles = np.zeros(3)

        for dof, angle in zip(self.dof, motion):
            if dof == 'rx':
                euler_angles[0] = angle
            elif dof == 'ry':
                euler_angles[1] = angle
            elif dof == 'rz':
                euler_angles[2] = angle

        euler_angles = np.deg2rad(euler_angles)
        self.matrix = self.Cinv @ rotation_matrix(*euler_angles) @ self.C @ self.parent.matrix
        self.coordinate = self.parent.coordinate + self.length * self.matrix @ self.direction

class RootJoint(Joint):
    '''
    represent the root joint
    '''
    def __init__(self):
        Joint.__init__(
            self,
            name='root',
            direction=np.zeros((3, 1)),
            length=0,
            axis=np.zeros(3),
            dof=[],
            limits=[]
        )

        self.step(np.zeros(6))

    def step(self, motion: np.ndarray):
        assert motion.shape == (6,)

        coordinate = motion[:3]
        rotation = motion[3:]

        self.coordinate = np.reshape(coordinate, (3, 1))
        rotation = np.deg2rad(rotation)
        self.matrix = self.C @ rotation_matrix(*rotation) @ self.Cinv

class Skeleton:
    '''
    skeleton
    '''
    def __init__(self, root: Joint):
        '''
        Arguments:
        ---
        - root: root of graph
        '''
        self._get_compute_sequence(root)

    def _get_compute_sequence(self, root: RootJoint):
        '''
        get compute sequence by topological ordering (DFS)
        '''
        self.compute_seq: list[Joint] = []
        self.joint_dict: dict[Joint, int] = dict()

        stack = [root]
        
        while stack:
            joint = stack.pop()

            self.joint_dict[joint] = len(self.compute_seq)
            self.compute_seq.append(joint)

            for c in joint.children:
                stack.append(c)

    def set_motion(self, motion: dict[str, any]):
        for joint in self.compute_seq:
            joint.step(motion.get(joint.name, np.zeros(3)))

    def set_coords(self, coords: np.ndarray):
        for i, joint in enumerate(self.compute_seq):
            joint.coordinate = coords[i, :].T

    def index2joint(self, idx: int) -> Joint:
        '''
        index to joint
        '''
        return self.compute_seq[idx]
        
    def joint2index(self, joint: Joint) -> int:
        '''
        joint to index
        '''
        return self.joint_dict[joint]

    def get_coords(self, path: str) -> np.ndarray:
        '''
        Arguments:
        ---
        - path: path to amc

        Returns:
        ---
        - coords[T, V, 3]: T = no. of frames, V = no. of joints
        '''
        motions = [*AMCParser().parse(path)]

        coords = np.zeros((len(motions), len(self.compute_seq), 3))

        for frame, motion in motions:
            self.set_motion(motion)

            for i, joint in enumerate(self.compute_seq):
                coords[frame - 1, i, :] = to_meter(joint.coordinate.T)

        return coords
    
    def plot(self):
        '''
        plot the skeleton with `matplotlib.pyplot`
        '''
        fig = plt.figure()
        ax = plt.axes(projection='3d')

        for joint in self.compute_seq:
            c = joint.coordinate
            #c = to_meter(c)
            ax.plot(c[2], c[0], c[1], 'b.')

            if joint.parent:
                p = joint.parent.coordinate
                #p = to_meter(p)

                ax.plot(
                    [c[2], p[2]],
                    [c[0], p[0]],
                    [c[1], p[1]],
                    'r'
                )

        return fig

class ASFParser:
    '''
    .asf file parser
    '''
    Idle = -1
    Root = 0
    BoneData = 1
    Hierarchy = 2
    Units = 3

    def __init__(self):
        self.state: int = ASFParser.Idle
        self.currentLine: str = None
        self.joints: dict[str, Joint] = None

    def parse(self, path: str) -> Skeleton:
        '''
        parse file in path

        Arguments:
        ---
        - path: path to `.asf` file

        Returns:
        ---
        - skeleton
        '''
        self.joints: dict[str, Joint] = dict()
        self.state = ASFParser.Idle
        self.file = open(path, 'r')
        self.currentLine = ''

        try:
            while self.currentLine is not None:
                if self.state == ASFParser.Root:
                    self._read_root()
                elif self.state == ASFParser.BoneData:
                    self._read_joints()
                elif self.state == ASFParser.Hierarchy:
                    self._read_hierarchy()
                else:
                    self._consume()

            return Skeleton(self.joints['root'])
        finally:
            self.file.close()
            self.file = None
            self.currentLine = None
            
            self.state = ASFParser.Idle
            self.joints = None

    def _read_root(self):
        '''
        read root section
        '''
        assert self.state == ASFParser.Root

        self.joints['root'] = RootJoint()

        while True:
            self._consume()
            if self.state != ASFParser.Root: break

    def _read_joints(self):
        '''
        read bonedata section
        '''
        assert self.state == ASFParser.BoneData

        while True:
            ln = self._consume()

            if self.state != ASFParser.BoneData: break
            assert ln == 'begin'

            # read one joint
            name: str = ''
            length: float = 0
            direction = np.zeros((3, 1))
            axis = np.zeros(3)

            dof: list[str] = []
            limits: list[tuple[float, float]] = []
            
            while True:
                ln = self._consume(expected_state=ASFParser.BoneData)
                if ln == 'end': break

                if ln.startswith('name'):
                    ln = ln.split(' ')
                    assert len(ln) == 2
                    name = ln[1]

                elif ln.startswith('length'):
                    ln = ln.split(' ')
                    assert len(ln) == 2
                    length = float(ln[1])

                elif ln.startswith('dof'):
                    ln = ln.split(' ')
                    dof = ln[1:]

                elif ln.startswith('axis'):
                    ln = ln.split(' ')
                    axis[0] = float(ln[1])
                    axis[1] = float(ln[2])
                    axis[2] = float(ln[3])

                elif ln.startswith('direction'):
                    ln = ln.split(' ')
                    direction[0] = float(ln[1])
                    direction[1] = float(ln[2])
                    direction[2] = float(ln[3])

                elif ln.startswith('limits'):
                    ln = ln.split(' ')

                    limits.append((
                        float(ln[1][1:]),
                        float(ln[2][:-1])
                    ))

                    for _ in range(1, len(dof)):
                        ln = self._consume(expected='(', expected_state=ASFParser.BoneData)
                        ln = ln.split(' ')

                        limits.append((
                            float(ln[0][1:]),
                            float(ln[1][:-1])
                        ))

            self.joints[name] = Joint(
                name=name,
                direction=direction,
                length=length,
                axis=axis,
                dof=dof,
                limits=limits
            )

    def _read_hierarchy(self):
        '''
        read hierarchy section
        '''
        assert self.state == ASFParser.Hierarchy
        self._consume('begin', ASFParser.Hierarchy)

        while True:
            ln = self._consume()

            if self.state != ASFParser.Hierarchy: break

            if ln == 'end':
                self._consume()
                break

            parent, *children = ln.split(' ')
            parent = self.joints[parent]

            for child in children:
                child = self.joints[child]

                parent.children.append(child)
                child.parent = parent

    def _consume(self, expected: str = None, expected_state: int = None) -> str:
        '''
        read one line from file and change state of parser

        Arguments:
        ---
        - expected: expected output
        - expected_state: expected state after read

        Returns:
        ---
        - line
        '''
        ln: str = None
        self.currentLine = None
        
        while (ln := self.file.readline()):
            ln = ln.strip()
            self.currentLine = ln
            if not ln: continue

            # check comment line
            if not ln.startswith('#'): break

        # check expected
        assert expected is None or (ln and ln.startswith(expected))

        if ln is not None and ln.startswith(':'):
            new_state: int = {
                'root': ASFParser.Root,
                'bonedata': ASFParser.BoneData,
                'hierarchy': ASFParser.Hierarchy
            }.get(ln.split(' ')[0][1:], self.state)

            assert expected_state is None or new_state == expected_state
            self.state = new_state
        
        return ln

class AMCParser:
    '''
    .amc file parser
    '''
    def __init__(self):
        self.file = None
        self.currentLine: str = None

    def parse(self, path: str):
        '''
        parse .amc file in path
        '''
        self.file = open(path, 'r')

        try:
            self.currentLine = ''

            while self.currentLine is not None:
                if self.currentLine.isnumeric():
                    frame = int(self.currentLine)
                    motion = self._read_frame()
                    yield frame, motion
                else:
                    self._consume()
        finally:
            self.file.close()
            self.file = None
            self.currentLine = None

    def _read_frame(self):
        '''
        read one frame

        Returns:
        ---
        - motion
        '''
        motions: dict[str, np.ndarray] = dict()

        while True:
            self._consume()

            if not self.currentLine or self.currentLine.isnumeric():
                break

            name, *rest = self.currentLine.split(' ')
            motions[name] = np.array([float(x) for x in rest], dtype=float)

        return motions

    def _consume(self):
        '''
        consume one line
        '''
        ln: str = None
        self.currentLine = None

        while (ln := self.file.readline()):
            ln = ln.strip()
            self.currentLine = ln
            if not ln: continue

            if not ln.startswith('#'): break

        return ln

def preprocess(inputDir: str, outputDir: str):
    '''
    preprocess dataset

    Arguments:
    ---
    - inputDir: input directory
    - outputDir: output directory
    '''
    # delete all files in outputDir
    if os.path.exists(outputDir):
        rmtree(outputDir)

    os.mkdir(outputDir)

    paths = os.listdir(inputDir)

    #region parse .asf
    skeleton: Skeleton = None

    for filename in paths:
        if filename.endswith('.asf.txt'):
            skeleton = ASFParser().parse(f'{inputDir}/{filename}')

            # save adjacency matrix
            adj_mat = np.zeros((len(skeleton.compute_seq), len(skeleton.compute_seq)), dtype=int)

            for joint in skeleton.compute_seq:
                pidx = skeleton.joint2index(joint)

                for children in joint.children:
                    cidx = skeleton.joint2index(children)
                    adj_mat[pidx, cidx] = 1
                    adj_mat[cidx, pidx] = 1

            prefix = filename.removesuffix('.asf.txt')
            torch.save(torch.from_numpy(adj_mat), f'{outputDir}/{prefix}.adj_mat.pt')
            break

    if not skeleton:
        raise Exception(f'.asf file not found in {inputDir}')
    #endregion
    
    #region parse .amc
    for filename in paths:
        if not filename.endswith('.amc.txt'):
            continue

        # parse file
        coords = skeleton.get_coords(f'{inputDir}/{filename}')
        
        prefix = filename.removesuffix('.amc.txt')
        torch.save(torch.from_numpy(coords), f'{outputDir}/{prefix}.pt')
    #endregion

DefaultInputDir = 'raw'
DefaultOutputDir = 'pt'

if __name__ == '__main__':
    preprocess(DefaultInputDir, DefaultOutputDir)