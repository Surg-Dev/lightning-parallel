#include <iostream>

#include "gvdb.h"
using namespace nvdb;

VolumeGVDB gvdb;

void init()
{
    int devid = -1;
    gvdb.SetDebug(true);
    gvdb.SetVerbose(true);
    gvdb.SetProfile(false, true);
    gvdb.SetCudaDevice(devid);
    gvdb.Initialize();

    gvdb.Configure(3, 3, 3, 3, 5);

    // Create one channel (phi) of type float
    gvdb.AddChannel(0, T_FLOAT, 1);
    // gvdb.VoxelizeNode()
}

int sample_main(int argc, const char **argv)
{
    // Sample sample_obj;
    // return sample_obj.run ( "NVIDIA(R) GVDB Voxels - gSprayDeposit", "spraydep", argc, argv, 1024, 768, 4, 5, 100 );
    init();
    std::cout << "Init successful" << std::endl;
    exit(0);
}

void sample_print(int argc, char const *argv)
{
}
