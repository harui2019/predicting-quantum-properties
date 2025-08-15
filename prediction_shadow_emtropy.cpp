//
// This code is created by Hsin-Yuan Huang (https://momohuang.github.io/).
// For more details, see the accompany paper:
//  "Predicting Many Properties of a Quantum System from Very Few Measurements".
//
#include <stdio.h>
#include <cmath>
#include <vector>
#include <sys/time.h>
#include <string>
#include <string.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cassert>
#include <utility>
#include <algorithm>

using namespace std;

int system_size = -1;
int number_of_observables;

double renyi_sum_of_binary_outcome[100000000];
double renyi_number_of_outcomes[100000000];

//
// The following function reads the file: subsystem_file_name
// and updates [subsystems].
//
vector<vector<int> > subsystems; // subsystems to predict entropy
void read_all_subsystems(char* subsystem_file_name){
    ifstream subsystem_fstream;
    subsystem_fstream.open(subsystem_file_name, ifstream::in);

    if(subsystem_fstream.fail()){
        fprintf(stderr, "\n====\nError: the input file \"%s\" does not exist.\n====\n", subsystem_file_name);
        exit(-1);
    }

    // Read in the system size
    int system_size_subsystem;
    subsystem_fstream >> system_size_subsystem;
    if(system_size == -1) system_size = system_size_subsystem;

    // Read in the local observables line by line
    string line;
    int observable_counter = 0;
    while(getline(subsystem_fstream, line)){
        if(line == "\n" || line == "") continue;
        istringstream single_line_stream(line);

        int k_local;
        single_line_stream >> k_local;

        vector<int> ith_subsystem;

        for(int k = 0; k < k_local; k++){
            int position_of_the_qubit;
            single_line_stream >> position_of_the_qubit;
            ith_subsystem.push_back(position_of_the_qubit);
        }

        subsystems.push_back(ith_subsystem);
    }
    subsystem_fstream.close();

    return;
}

//
// The following function reads the file: measurement_file_name
// and updates [observables] and [observables_acting_on_ith_qubit]
//
vector<vector<int> > measurement_pauli_basis;
vector<vector<int> > measurement_binary_outcome;
void read_all_measurements(char* measurement_file_name){
    ifstream measurement_fstream;
    measurement_fstream.open(measurement_file_name, ifstream::in);

    if(measurement_fstream.fail()){
        fprintf(stderr, "\n====\nError: the input file \"%s\" does not exist.\n====\n", measurement_file_name);
        exit(-1);
    }

    // Read in the system size
    int system_size_measurement;
    measurement_fstream >> system_size_measurement;
    if(system_size == -1) system_size = system_size_measurement;
    if(system_size_measurement != system_size){
        fprintf(stderr, "\n====\nError: the system size do not match.\n====\n");
        exit(-1);
    }

    // Read in the measurements line by line
    string line;
    int measurement_counter = 0;
    while(getline(measurement_fstream, line)){
        if(line == "\n" || line == "") continue;
        istringstream single_line_stream(line);

        vector<int> empty_list;
        measurement_pauli_basis.push_back(empty_list);
        measurement_binary_outcome.push_back(empty_list);
        for(int ith_qubit = 0; ith_qubit < system_size; ith_qubit++){
            char pauli[10];
            int binary_outcome;
            single_line_stream >> pauli >> binary_outcome;
            assert(binary_outcome == 1 || binary_outcome == -1);

            measurement_pauli_basis[measurement_counter].push_back(pauli[0] - 'X');
            measurement_binary_outcome[measurement_counter].push_back(binary_outcome);
        }

        measurement_counter ++;
    }
}


int main(int argc, char* argv[]){

    //
    // Running the prediction of entanglement entropy
    //
    read_all_measurements(argv[2]);
    read_all_subsystems(argv[3]);

    printf("purity   entropy\n");
    for(int s = 0; s < (int)subsystems.size(); s++){
        int subsystem_size = (int)subsystems[s].size();


        for(int c = 0; c < (1 << (2 * subsystem_size)); c++){
            renyi_sum_of_binary_outcome[c] = 0;
            renyi_number_of_outcomes[c] = 0;
        }

        for(int t = 0; t < (int)measurement_pauli_basis.size(); t++){
            long long encoding = 0, cumulative_outcome = 1;

            renyi_sum_of_binary_outcome[0] += 1;
            renyi_number_of_outcomes[0] += 1;

            // Using gray code iteration over all 2^n possible outcomes
            for(long long b = 1; b < (1 << subsystem_size); b++){
                long long change_i = __builtin_ctzll(b);
                long long index_in_original_system = subsystems[s][change_i];

                cumulative_outcome *= measurement_binary_outcome[t][index_in_original_system];
                encoding ^= (measurement_pauli_basis[t][index_in_original_system] + 1) << (2LL * change_i);

                renyi_sum_of_binary_outcome[encoding] += cumulative_outcome;
                renyi_number_of_outcomes[encoding] += 1;
            }
        }

        int level_cnt[2 * subsystem_size], level_ttl[2 * subsystem_size];
        for(int i = 0; i < subsystem_size + 1; i++){
            level_cnt[i] = 0;
            level_ttl[i] = 0;
        }

        for(long long c = 0; c < (1 << (2 * subsystem_size)); c++){
            int nonId = 0;
            for(int i = 0; i < subsystem_size; i++){
                nonId += ((c >> (2 * i)) & 3) != 0;
            }
            if(renyi_number_of_outcomes[c] >= 2)
                level_cnt[nonId] ++;
            level_ttl[nonId] ++;
        }

        double predicted_entropy = 0;
        for(long long c = 0; c < (1 << (2 * subsystem_size)); c++){
            if(renyi_number_of_outcomes[c] <= 1) continue;

            int nonId = 0;
            for(int i = 0; i < subsystem_size; i++)
                nonId += ((c >> (2 * i)) & 3) != 0;

            predicted_entropy += ((double)1.0) / (renyi_number_of_outcomes[c] * (renyi_number_of_outcomes[c] - 1)) * (renyi_sum_of_binary_outcome[c] * renyi_sum_of_binary_outcome[c] - renyi_number_of_outcomes[c]) / (1LL << subsystem_size) * level_ttl[nonId] / level_cnt[nonId];
        }

        printf("%f %f\n", predicted_entropy, -1.0 * log2(min(max(predicted_entropy, 1.0 / pow(2.0, subsystem_size)), 1.0 - 1e-9)));
    }
}
