import argparse
import os.path
import numpy
import time
import math
from multiprocessing import Pool

from . import mesh_slice
from . import stl_reader
from . import perimeter

mesh = []
bounding_box = []

def read_and_reshape_stl(inputFilePath, resolution):
    """
    Read stl file and reshape mesh
    """
    mesh = list(stl_reader.read_stl_verticies(inputFilePath))
    (scale, shift, bounding_box) = mesh_slice.calculateScaleAndShift(mesh, resolution)
    mesh = list(mesh_slice.scaleAndShiftMesh(mesh, scale, shift))
    return (mesh, bounding_box)


def voxelize_parallel(K_chunks):

    mesh = K_chunks[2]

    bounding_box = K_chunks[3]

    prepixel = numpy.zeros((K_chunks[1] - K_chunks[0], bounding_box[0], bounding_box[1]), dtype=numpy.int8)

    for height in range(K_chunks[0], K_chunks[1]):

        # if thread_id == 0:
        #     print("processing layer {}", height + 1)
        lines = mesh_slice.toIntersectingLines(mesh, height)
        # prepixel = voxels[height]
        perimeter.linesToVoxels(lines, prepixel[height - K_chunks[0]])

    return prepixel


def voxelize(mesh, bounding_box, nprocs):
    """
    Voxelize a mesh with a given bounding box
    """

    # print(bounding_box)
    # voxels = numpy.zeros((bounding_box[2], bounding_box[0], bounding_box[1]), dtype=numpy.int8)

    start = time.time()

    K_chunks = [[n * bounding_box[2] // nprocs, (n + 1) * bounding_box[2] // nprocs, mesh, bounding_box] for n in range(nprocs)]

    pool = Pool(processes=nprocs)

    outputs = pool.map(voxelize_parallel, K_chunks)

    voxels = numpy.concatenate(outputs, axis=0)

    print("time spent for voxelization mesh :", time.time() - start, "s")

    voxels = numpy.transpose(voxels, (2, 1, 0))

    return voxels


def to_ascii(voxels, bounding_box, outputFilePath):
    """
    Create an ascii file and store the voxel array
    """
    output = open(outputFilePath, 'w')
    for z in bounding_box[2]:
        for x in bounding_box[0]:
            for y in bounding_box[1]:
                if voxels[z][x][y]:
                    output.write('%s %s %s\n' % (x, y, z))
    output.close()


if __name__ == '__main__':
    # parse cli args
    parser = argparse.ArgumentParser(description='Convert STL files to voxels')
    parser.add_argument('input', nargs='?')
    parser.add_argument('output', nargs='?')
    parser.add_argument('resolution', nargs='?', type=int)
    args = parser.parse_args()
    # read and rescale
    mesh, bounding_box = read_and_reshape_stl(args.input, args.resolution)
    # create voxel array
    voxels, bounding_box = voxelize(mesh, bounding_box)
    #store the result
    to_ascii(voxels, bounding_box, args.output)
