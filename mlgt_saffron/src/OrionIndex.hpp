#ifndef H_7D4E5B12_A8F9_4C3D_9E71_2B6F8A4D1C90
#define H_7D4E5B12_A8F9_4C3D_9E71_2B6F8A4D1C90



#include "headers.hpp"
#include <algorithm>
#include <execution>
#include <fstream>
#include <stdexcept>
#include <parallel/algorithm>
#include "ProgressBar.hpp"
#include <atomic>
#include <tuple>

struct PostingRange{
    uint start_index;
    uint item_count;
};
struct HashNode
{
    unordered_map<uint, PostingRange> postings;
    // vector<vector<bool>> bloomFilter;
};


/**
 * @brief A True Inverted Index that will store hashes and the global ids of the items that set them as follows
 * 
 * Hash Function range is 0 to R
 * We have L such hash functions
 * The inverted index will store an array of size R which will have the hash_vals and a pointer to an flattened hashmap
 * The hashmap key = hash_function + h_val * num_hashes_
 * How to access the elements: hash_buckets_[pool_id].postings[key]
 * 
 */
class OrionIndex {
public:
    vector<HashNode> hash_buckets_; 
    uint hash_range_;
    uint num_hashes_;
    uint threshold_;
    uint num_pools_;
    uint num_shards_;
    vector<vector<uint>> doc_index_; 

    /**
     * @brief Empty constructor for OrionIndex.
     */
    OrionIndex() : hash_range_(0), num_hashes_(0), threshold_(0), num_pools_(0), num_shards_(32) {}
    
    /**
     * @brief Construct a new Orion Index with specified parameters.
     * @param num_pools_ The number of pools.
     * @param num_hashes The number of hashes (LSH functions) per item.
     * @param threshold The number of matching hashes required for a query match.
     */
    OrionIndex(uint num_pools, uint num_hashes, uint threshold) 
        : num_pools_(num_pools), num_hashes_(num_hashes), threshold_(threshold),  num_shards_(32) {
        hash_buckets_.resize(num_pools_);
        
    }

    /**
     * @brief Generates Key for the Hashmap
     *  
     * @param hash_index The id of Hash Function
     * @param h_val The hash value
     */
    
    inline uint getPoolHashKey(uint hash_index, uint h_val) const {
        uint key = hash_index + h_val * num_hashes_;
        return key;
    }

    /**
     * @brief Builds the inverted index from given item hashes.
     *  
     * @param all_hashes A 2D vector where all_hashes[i] contains the hashes for item i.
     * @param item_indices A 2D vector where item_indices[pool_id] contains the global ids of the items in that pool
     */
    void build( vector<vector<uint>>& all_hashes, const vector<vector<uint>>& item_indices) {
        if (all_hashes.empty() || item_indices.empty()) return;
        
        num_hashes_ = all_hashes[0].size();
        num_pools_ = item_indices.size();
        
        hash_buckets_.clear();
        hash_buckets_.resize(num_pools_);
        doc_index_.clear();

        uint num_features = all_hashes.size();
        uint L =  ceil(log2(num_features));

        uint64_t total_postings = POOLS_PER_ITEM * all_hashes.size() * num_hashes_ * L * 2;
        uint64_t postings_per_shard = ((total_postings + num_shards_ - 1) / num_shards_) * 103 / 100;
        doc_index_.resize(num_shards_);
        for(uint shard = 0; shard < num_shards_; shard++){
            doc_index_[shard].reserve(postings_per_shard);
        }


        // doc_index_.reserve(POOLS_PER_ITEM * all_hashes.size() * num_hashes_ * L * 3);

        struct Entry {
            uint32_t key;
            uint32_t item_id;
        };

        ProgressBar progress(num_pools_);
        std::atomic<uint64_t> completed_pools{0};
        #pragma omp parallel for num_threads(num_shards_)
        for(uint shard = 0; shard < num_shards_; shard++){
            for(uint pool_id = shard; pool_id < num_pools_;  pool_id += num_shards_ ){
                
                std::vector<Entry> entries;
                entries.reserve(39000000);
                for(uint item : item_indices[pool_id]){
                    
                    for(uint h = 0;  h < num_hashes_; h++){
                        Entry x;
                        uint h_val = all_hashes[item][h];
                        uint key = getPoolHashKey(h, h_val);
                        x.key = key;
                        x.item_id = item;
                        entries.push_back(x);

                    }

                    

                }


                // sort entries by hash_val and then among same hash_vals by key
                __gnu_parallel::sort( entries.begin(), entries.end(),
                    [](const Entry& a, const Entry& b) {
                        return a.key < b.key;
                    });

                
                for(int i = (int)entries.size() - 1 ; i >= 0; ){
                    uint key = entries[i].key;
                    // int j =  i - 1;
                    // uint shard = pool_id % num_shards_;
                    uint start_index = doc_index_[shard].size();
                    uint item_count = 1;
                    PostingRange info;
                    info.start_index = start_index;
                    doc_index_[shard].push_back(entries[i].item_id);
                    i--;
                    while(i >= 0 && key == entries[i].key){
                        item_count++;
                        doc_index_[shard].push_back(entries[i].item_id);
                        i--;
                        
                    }
                

                    info.item_count = item_count;
                    hash_buckets_[pool_id].postings[key] = info;

                }
                uint64_t done = completed_pools.fetch_add(1) + 1;
                progress.update(done);

            }
        
        }
    }


    void debug_lookup(uint pool, uint key) {
        auto it = hash_buckets_[pool].postings.find(key);
        if (it == hash_buckets_[pool].postings.end())
            cout << "NOT FOUND\n";
        else
            cout << it->second.start_index << " "
                << it->second.item_count << endl;
    }


    /**
     * @brief It returns the result of a test as 0 (Negative) or 1 (Positive) and the defective item identified and length of postings traversed.
     * IMPORTANT
     * Please note that if the test result is negaative the uint value is set by default to 0.
     * @param query_hashes The pre-computed hashes of the query vector.
     * @param pool_index Index of the pool being evaluated.
     * @return tuple<bool, uint, uint> Test Result and if positive also has the global_id of the positive item and the length of postings list traversed.
     */
    inline std::tuple<bool, uint, uint> get_matches(const vector<uint> &query_hashes, uint pool_index) const {
        
        uint postings_traversed = 0;
        if (num_hashes_ == 0 ) return {false, 0, postings_traversed};
        
                    
        unordered_map<uint, uint> counts;
        uint hash_misses = 0;
        uint shard = pool_index % num_shards_; 


        for(uint h = 0; h < num_hashes_ ; h++){
            uint q_h = query_hashes[h];
            uint key = getPoolHashKey(h, q_h);

            // Checking if query hash can match threshold
            if(hash_misses > num_hashes_ - threshold_){
                return {false, 0, postings_traversed};
            }
            
            // Skip if Query Hash doesn't exist in Index
            auto it = hash_buckets_[pool_index].postings.find(key);
            if(it == hash_buckets_[pool_index].postings.end()){
                hash_misses++;
                continue;
            }

            uint start_index = it->second.start_index;
            uint item_count = it->second.item_count;
            // it->second is just hash_buckets_[pool_index].postings[key] 

  

            for( uint i = start_index; i < start_index + item_count; i++  ){
                uint global_id = doc_index_[shard][i];
                postings_traversed++;
                if(++counts[global_id] >= threshold_){
                    // cout<<"Positive item_id : "<<global_id<<endl<<"Pool Index : "<<pool_index<<endl;
                    return {true, global_id, postings_traversed};
                }
            }              
        }
        return {false, 0, postings_traversed};        
    }

    /**
     * @brief It returns the result of a test as 0 (Negative) or 1 (Positive) and all defective items identified and length of postings traversed.
     * IMPORTANT
     * Please note that if the test result is negaative the uint value is set by default to 0.
     * @param query_hashes The pre-computed hashes of the query vector.
     * @param pool_index Index of the pool being evaluated.
     * @return tuple<bool, uint, uint> Test Result and if positive also has the global_id of the positive item and the length of postings list traversed.
     */
    inline std::tuple<bool, set<uint>, uint> get_full_matches(const vector<uint> &query_hashes, uint pool_index) const {
        
        uint postings_traversed = 0;       
        set<uint> all_defectives;
        if (num_hashes_ == 0 ) return {false, all_defectives, postings_traversed};
        
                    
        unordered_map<uint, uint> counts;
        uint hash_misses = 0;
        uint shard = pool_index % num_shards_; 


        for(uint h = 0; h < num_hashes_ ; h++){
            uint q_h = query_hashes[h];
            uint key = getPoolHashKey(h, q_h);

            // Checking if query hash can match threshold
            if(hash_misses > num_hashes_ - threshold_){
                return {false, all_defectives, postings_traversed};
            }
            
            // Skip if Query Hash doesn't exist in Index
            auto it = hash_buckets_[pool_index].postings.find(key);
            if(it == hash_buckets_[pool_index].postings.end()){
                hash_misses++;
                continue;
            }

            uint start_index = it->second.start_index;
            uint item_count = it->second.item_count;
            // it->second is just hash_buckets_[pool_index].postings[key] 

  

            for( uint i = start_index; i < start_index + item_count; i++  ){
                uint global_id = doc_index_[shard][i];
                postings_traversed++;
                if(++counts[global_id] == threshold_){
                    // cout<<"Positive item_id : "<<global_id<<endl<<"Pool Index : "<<pool_index<<endl;
                    all_defectives.insert(global_id);
                }
            }              
        }
       
        return {!all_defectives.empty(), all_defectives, postings_traversed};        
    }
 
    /**
     * @brief Returns the number of hash functions the index expects.
     * @return uint Hash count.
     */
    // inline uint num_hashes() const { return num_hashes_; }
};



#endif /* H_7D4E5B12_A8F9_4C3D_9E71_2B6F8A4D1C90 */
