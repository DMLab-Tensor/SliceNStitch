#include "loader.hpp"
#include "tensor.hpp"
#include "tensorStream.hpp"
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <string>

int main(int argc, char* argv[])
{
    /* Second argument should be the path of config file */
    if (argc < 2) {
        std::cerr << "Error: You have to load config file." << std::endl;
        return EXIT_FAILURE;
    }

    const std::string configPath = argv[1];

    /* Parse config file */
    const Config config = Config(configPath);
    std::cout << "Info: Load configuration successfully" << std::endl;

    /* Load tensor */
    DataStream* paperX = config.loadDataStream();
    std::cout << "Info: Load data successfully" << std::endl;

    /* Generate TensorStream */
    TensorStream* ts = generateTensorStream(*paperX, config);
    std::cout << "Info: Initialize successfully" << std::endl;

    /* Generate files, data structures for saving options and the results */
    std::ofstream outFile(config.outputPath());
    outFile << "-----Options-----\n";
    std::ifstream configFile(configPath);
    outFile << configFile.rdbuf() << "\n\n";
    outFile << "-----Test Results-----\n";
    outFile << "RMSE\tFitness\n";

    double rmse, fitness;

    /* Run */
    std::cout << "Info: Start running the algorithm" << std::endl;
    rmse = ts->rmse();
    fitness = ts->fitness();
    //fitness = ts->fitness_latest();

    outFile << rmse << "\t" << fitness << "\n";
    printf("[Initial]\tRMSE: %f,\tFitness: %f\n", rmse, fitness);

    int udPeriod = config.updatePeriod();

    long long currTime = config.startTime() + config.unitNum() * config.unitSize() - 1;

    int numUpdate = 0;

    for (int epoch = 1; epoch <= config.numEpoch(); ++epoch) {
        // Update datastream
        paperX->update(++currTime);

        while (true) {
            auto e = paperX->pop();
            if (e == nullptr) {
                break;
            }

            // Update tensor
            ts->updateTensor(*e);

            // Update factor matrices
            if (epoch % udPeriod == 0) {
                numUpdate++;
                ts->updateFactor();
            }

            // Print its original coordinate and the reconstruction error
            /*
            if (e->newUnitIdx == config.unitNum() - 1) {
                const int numMode = config.numMode();

                std::vector<int> coord(numMode);
                for (int m = 0; m < numMode - 1; ++m) {
                    coord[m] = e->nonTempCoord[m];
                }
                coord[numMode - 1] = e->newUnitIdx;

                printf("Error: at (");
                for (int m = 0; m < numMode - 1; ++m) {
                    printf("%d, ", coord[m]);
                }
                printf("%lld): %f\n", currTime, ts->error(coord));
            }
            */

            /*
            rmse = ts->rmse();
            fitness = ts->fitness();
            outFile << rmse << "\t" << fitness << "\n";
            printf("[Epoch %d] - event\tRMSE: %f,\tFitness: %f\n", epoch, rmse, fitness);
            */
        }

        // Check error
        if (epoch % config.checkPeriod() == 0) {
            ts->updateAtA();

            rmse = ts->rmse();
            fitness = ts->fitness();
            //fitness = ts->fitness_latest();
            outFile << rmse << "\t" << fitness << "\n";
            printf("[Epoch %d]\tRMSE: %f,\tFitness: %f\n", epoch, rmse, fitness);
        }
    }

    const double elapsedTime = ts->elapsedTime();

    outFile << "\n-----The Total Number of Updates-----\n";
    outFile << numUpdate << "\n";

    outFile << "\n-----Total Elapse Time-----\n";
    outFile << elapsedTime << " (sec)\n";

    outFile << "\n-----Elapse Time per each Update-----\n";
    outFile << elapsedTime / numUpdate << " (sec)\n";

    outFile.close();

    printf("Info: Finished (The total number of updates: %d)\n", numUpdate);
    printf("Total elapsed time: %f sec\n", elapsedTime);
    printf("Elapsed time per each update: %f sec\n", elapsedTime / numUpdate);

    /* Free */
    delete ts;
    delete paperX;

    return EXIT_SUCCESS;
}
