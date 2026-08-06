#ifndef E8F0D2A4_7B4C_4C8E_9A61_5D2F7A9B3E14
#define E8F0D2A4_7B4C_4C8E_9A61_5D2F7A9B3E14

#include "BloomHashFunction.hpp"
#include "OrionIndex.hpp"
#include <numeric>
#include <tuple>
#include "MaskMatrix.hpp"


class BruteForceHashSearch{
    public:
    // vector<vector<uint>> all_hashes_;
    BloomHashFunction shared_hasher_; 
    uint threshold_;
    uint num_hashes_;
    uint num_features_;
    vector<vector<bool>> all_hashes_;
    OrionIndex GlobalIndex_;

    BruteForceHashSearch( uint threshold, uint num_features, uint dimension, uint num_hashes, uint hash_bits, int debug, vector<vector<bool>> all_hashes)
    : 
      num_features_(num_features),
      shared_hasher_(dimension, num_hashes, hash_bits, threshold, debug),
      num_hashes_(num_hashes),
      threshold_(num_hashes - threshold),
      GlobalIndex_(num_hashes_, num_hashes_, threshold_),
      all_hashes_(all_hashes)


      
    {
       
        
        cout << "Threshold in Brute Index: " << threshold_ << endl;
        cout << "num_hashes in Brute Index: " << num_hashes_ << endl;

        vector<vector<uint>> uint_hashes(num_features_, vector<uint>(num_hashes_));
        #pragma omp parallel for collapse(2) schedule(static)
        for(uint i = 0; i < num_features_; i++){
            for(uint j = 0; j < num_hashes_; j++){
                uint_hashes[i][j] = 0;
                if(all_hashes_[i][j])
                    uint_hashes[i][j] = 1;

            }
        }
        
        
        // Items in pool
        vector<uint> items(num_features_);
        std::iota(items.begin(), items.end(), 0);
        vector<vector<uint>> all_items;
        all_items.push_back(items);
        cout<< "Building Global Inverted Index for Brute Force Search"<<endl;
        GlobalIndex_.build(uint_hashes, all_items);
        cout<<"GlobalIndex Built"<<endl;

        all_hashes_.clear();
        all_hashes_.shrink_to_fit();


    }


    /**
     * @brief Performs Brute Force Search using an Inverted Index for each bit.
     * 
     * @param query_arr The query vector (numpy array).
     * @return tuple of [ set<uint>, uint, double] which is the true defective items, postings_traversed, naive_search_time.
     */
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
        auto search_end = std::chrono::high_resolution_clock::now();
        double naive_search_time = std::chrono::duration<double>(search_end - search_start).count();

        return {bruteforce_defectives, postings_traversed, naive_search_time};
    }



};

#endif // E8F0D2A4_7B4C_4C8E_9A61_5D2F7A9B3E14