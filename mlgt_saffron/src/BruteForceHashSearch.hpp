#ifndef E8F0D2A4_7B4C_4C8E_9A61_5D2F7A9B3E14
#define E8F0D2A4_7B4C_4C8E_9A61_5D2F7A9B3E14

#include "BloomHashFunction.hpp"
#include "OrionIndex.hpp"
#include <numeric>
#include <tuple>

class BruteForceHashSearch{
    public:
    vector<vector<uint>> all_hashes_;
    BloomHashFunction shared_hasher_; 
    uint threshold_;
    uint num_hashes_;
    uint num_features_;
    OrionIndex GlobalIndex_;

    BruteForceHashSearch(uint threshold, uint num_features, uint dimension, uint num_hashes, uint hash_bits, int debug)
    : 
      threshold_(threshold),
      num_features_(num_features),
      shared_hasher_(dimension, num_hashes, hash_bits, threshold, debug),
      num_hashes_(num_hashes),
      GlobalIndex_(1, num_hashes_, threshold_)


      
    {
        cout << "Loading precomputed hashes for Naive Search..." << endl;
        all_hashes_.resize(num_features_);

        
        std::ifstream file("/home/adnan/projects/MLGTSaffron/results/all_hashes_2.csv");
        if (!file.is_open()) {
            std::cerr << "Failed to open file!" << std::endl;
            std::exit(1);
        }

        std::string line;
        for (uint i = 0; i < num_features_ && std::getline(file, line); i++) {
            std::stringstream ss(line);
            std::string value;

            while (std::getline(ss, value, ',')) {
               all_hashes_[i].push_back(std::stoul(value));
            }
        }
        cout << "Finished loading hashes for Naive Search." << endl;
        
        // Items in pool
        vector<uint> items(num_features_);
        std::iota(items.begin(), items.end(), 0);
        vector<vector<uint>> all_items;
        all_items.push_back(items);
        cout<< "Building Global Inverted Index for Brute Force Search"<<endl;
        GlobalIndex_.build(all_hashes_, all_items);
        cout<<"GlobalIndex Built"<<endl;


    //   num_features_ = all_hashes.size();
    //   num_hashes_ =all_hashes_.empty() ? 0 :all_hashes_[0].size();
        all_hashes_.clear();
        all_hashes_.shrink_to_fit();


    }

    std::tuple<set<uint>, uint, double> bruteSearch(pybind11::array_t<float> query_arr){

        auto search_start = std::chrono::high_resolution_clock::now();

        Eigen::Map<const Eigen::VectorXf> q_raw(query_arr.data(), query_arr.shape(0));
        Eigen::VectorXf query = q_raw;
        if (true) {
            float norm = query.norm();
            if (norm > 1e-9) query /= norm;
        }

        vector<uint> counts;
        counts.resize(num_features_, 0);

        vector<uint> query_hashes = shared_hasher_(query);


        
        auto [status, bruteforce_defectives, postings_traversed] = GlobalIndex_.get_full_matches(query_hashes, 0);

        // for(uint item_id = 0; item_id < num_features_; item_id++ ){
        //     for(uint h = 0; h < num_hashes_; h++){
        //         if(query_hashes[h] == all_hashes_[item_id][h]){
        //             counts[item_id]++;
        //         }
        //     }
        // }
        // for( uint item_id = 0; item_id < num_features_; item_id++){
        //     if(counts[item_id] >= threshold_){
        //         bruteforce_defectives.insert(item_id);
        //     }
        // }
        auto search_end = std::chrono::high_resolution_clock::now();
        double naive_search_time = std::chrono::duration<double>(search_end - search_start).count();

        return {bruteforce_defectives, postings_traversed, naive_search_time};
    }



};

#endif // E8F0D2A4_7B4C_4C8E_9A61_5D2F7A9B3E14