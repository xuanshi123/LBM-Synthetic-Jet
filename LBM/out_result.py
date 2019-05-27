import numpy as np
import os as os
from matplotlib import cm
import matplotlib.pyplot as plt
import math
from evtk.hl import gridToVTK


def output_result(lu, ts, ftemp, result_folder, t):
    if not os.path.exists(result_folder):
        os.mkdir(result_folder)
        print("Result Directory ", result_folder, " Created ")

    rho = np.sum(ftemp, axis=2)

    vx = ftemp[:, :, 1] - ftemp[:, :, 2] + ftemp[:, :, 7] - ftemp[:, :, 8] + ftemp[:, :, 10] - ftemp[:, :, 9] + ftemp[
        :, :, 11] - ftemp[:, :, 12] + ftemp[:, :, 16] - ftemp[:, :, 15]

    vy = ftemp[:, :, 3] - ftemp[:, :, 4] + ftemp[:, :, 7] - ftemp[:, :, 8] + ftemp[:, :, 9] - ftemp[:, :, 10] + ftemp[
        :, :, 13] - ftemp[:, :, 14] + ftemp[:, :, 18] - ftemp[:, :, 17]

    vz = ftemp[:, :, 5] - ftemp[:, :, 6] + ftemp[:, :, 11] - ftemp[:, :, 12] + ftemp[:, :, 13] - ftemp[:, :, 14] + ftemp[
        :, :, 15] - ftemp[:, :, 16] + ftemp[:, :, 17] - ftemp[:, :, 18]

    vx_p = np.divide(vx, rho, out=np.zeros_like(vx, dtype=np.float64), where=rho != 0) * lu / ts

    vy_p = np.divide(vy, rho, out=np.zeros_like(vy, dtype=np.float64), where=rho != 0) * lu / ts

    vz_p = np.divide(vz, rho, out=np.zeros_like(vz, dtype=np.float64), where=rho != 0) * lu / ts

    # cap = np.max(vy_p)
    # print(cap)
    # print(np.where(vy_p == cap))

    dpi_i = 100 if np.max(rho.shape) / 3 < 100 else round(np.max(rho.shape) / 150) * 50

    # plt.clf()
    # plt.title('Vx m/s')
    # plt.xlabel('x ')
    # plt.ylabel('y ')
    # cap = np.max(np.abs(vx_p))
    # plt.imshow(vx_p, cmap=cm.plasma, vmin=-cap, vmax=cap)
    # plt.colorbar()
    # plt.gca().invert_yaxis()
    # fname = result_folder + "/Vx time {:11.9f} ms.png".format(t * 1000)
    # plt.savefig(fname, dpi=dpi_i, bbox_inches='tight', pad_inches=0)
    #
    # plt.clf()
    # plt.title('Vy m/s')
    # plt.xlabel('x ')
    # plt.ylabel('y ')
    # cap = np.max(np.abs(vy_p))
    # plt.imshow(vy_p, cmap=cm.plasma, vmin=-cap, vmax=cap)
    # plt.colorbar()
    # plt.gca().invert_yaxis()
    # fname = result_folder + "/Vy time {:11.9f} ms.png".format(t * 1000)
    # plt.savefig(fname, dpi=dpi_i, bbox_inches='tight', pad_inches=0)

    # plt.clf()
    # plt.title('Vz m/s')
    # plt.xlabel('x ')
    # plt.ylabel('y ')
    # cap = np.max(np.abs(vz_p))
    # plt.imshow(vz_p, cmap=cm.plasma, vmin=-cap, vmax=cap)
    # plt.colorbar()
    # plt.gca().invert_yaxis()
    # fname = result_folder + "/Vz time {:11.9f} ms.png".format(t * 1000)
    # plt.savefig(fname, dpi=dpi_i, bbox_inches='tight', pad_inches=0)

    velocity_p = np.sqrt(vx_p * vx_p + vy_p * vy_p)
    plt.clf()
    plt.title('total velocity m/s')
    plt.xlabel('x ')
    plt.ylabel('y ')
    plt.imshow(velocity_p, cmap=cm.plasma)
    plt.colorbar()
    plt.gca().invert_yaxis()
    fname = result_folder + "/total velocity time {:11.9f} ms.png".format(t * 1000)
    plt.savefig(fname, dpi=dpi_i, bbox_inches='tight', pad_inches=0.01)

    plt.clf()
    plt.title('Vector Plot m/s')
    plt.xlabel('x ')
    plt.ylabel('y ')

    mag = np.sqrt(vx_p ** 2 + vy_p ** 2)
    Vx = np.divide(vx_p, mag, out=np.zeros_like(vx_p, dtype=np.float64), where=mag != 0)
    Vy = np.divide(vy_p, mag, out=np.ones_like(vy_p, dtype=np.float64), where=mag != 0)

    scale = int(round(np.max(Vx.shape) / 40))

    start = int(scale / 2)

    plt.quiver(Vx[start::scale, start::scale], Vy[start::scale, start::scale], mag[start::scale, start::scale],
               angles='xy', pivot='tail', headwidth=3, cmap=cm.plasma)

    plt.colorbar()
    fname = result_folder + "/vector plot {:11.9f} ms.png".format(t * 1000)
    plt.savefig(fname, dpi=350, bbox_inches='tight', pad_inches=0.01)


    # X = np.arange(0, lu * velocity_p.shape[0], lu, dtype='float32')
    # Y = np.arange(0, lu * velocity_p.shape[1], lu, dtype='float32')
    # Z = np.array([0])
    #
    # fname = result_folder + "/total velocity time {:11.9f} ms".format(t * 1000)
    #
    # velocity_p = velocity_p.reshape((velocity_p.shape[0], velocity_p.shape[1], 1))
    # gridToVTK(fname, X, Y, Z, pointData={"Total Velocity": velocity_p})
