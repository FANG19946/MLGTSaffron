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
#include "MaskMatrix.hpp"

struct PostingRange{
    uint start_index;
    uint item_count;
};
struct HashNode
{
    unordered_map<uint, PostingRange> postings;
    
};


/**
 * @brief A True Inverted Index that will store hashes and the global ids of the items that set them as follows
 * 
 * Hash Function range is 0 to R
 * We have L such hash functions
 * The inverted index will store an array of size num_masks_ which will have the mask_id and a pointer to an flattened hashmap
 * The hashmap key = hash_function + h_val * num_hashes_
 * How to access the elements: hash_buckets_[pool_id].postings[key] will return a PostingRange which 
 * contains the start_index and item_count which points to the sharded doc_index_, the corresponding postings can be obtained by
 * doc_index_[shard][start_idx] till doc_index_[shard][start_idx + item_count - 1].
 * 
 */
class OrionIndex {
public:
    vector<HashNode> hash_buckets_; 
    uint num_hashes_;
    uint threshold_;
    uint num_masks_;
    uint num_shards_;
    uint mask_size_;
    uint num_features_;
    vector<vector<uint>> doc_index_; 
    vector<uint8_t> visited_;


    /**
     * @brief Empty constructor for OrionIndex.
     */
    OrionIndex() :  num_hashes_(0), threshold_(0), num_masks_(0), num_shards_(32) {}
    
    /**
     * @brief Construct a new Orion Index with specified parameters.
     * @param num_masks_ The number of pools.
     * @param num_hashes The number of hashes (LSH functions) per item.
     * @param threshold The number of matching hashes required for a query match.
     */
    OrionIndex(uint num_features, uint num_masks, uint num_hashes, uint threshold) 
        : num_features_(num_features), num_masks_(num_masks), num_hashes_(num_hashes), threshold_(threshold),  num_shards_(32), visited_(num_features_, 0) {
        hash_buckets_.resize(num_masks_);
        
    }

    /**
     * @brief Generates Key for the Hashmap
     *  
     * @param hash_index The id of Hash Function
     * @param h_val The hash value
     */
    
    inline uint getPoolHashKey(uint hash_index, bool h_val) const {
       uint x = 0;
        if(h_val){
            x = 1;
       }
        uint key = hash_index + x * num_hashes_;
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
        num_masks_ = item_indices.size();
        
        hash_buckets_.clear();
        hash_buckets_.resize(num_masks_);
        doc_index_.clear();

        uint num_features = all_hashes.size();
        uint L =  ceil(log2(num_features));

        uint64_t total_postings = POOLS_PER_ITEM * all_hashes.size() * num_hashes_ * L * 2;
        uint64_t postings_per_shard = ((total_postings + num_shards_ - 1) / num_shards_) * 103 / 100;
        doc_index_.resize(num_shards_);
        for(uint shard = 0; shard < num_shards_; shard++){
            doc_index_[shard].reserve(postings_per_shard);
        }

        struct Entry {
            uint32_t key;
            uint32_t item_id;
        };

        ProgressBar progress(num_masks_);
        std::atomic<uint64_t> completed_pools{0};
        #pragma omp parallel for num_threads(num_shards_)
        for(uint shard = 0; shard < num_shards_; shard++){
            for(uint pool_id = shard; pool_id < num_masks_;  pool_id += num_shards_ ){
                
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

    /**
     * @brief Builds the inverted index from given item binary hashes and masks.
     *  
     * @param all_hashes A 2D vector where all_hashes[i] contains the binary hashes for item i.
     * @param sky_map A MaskMatrix which contains all the masks for the KHAN. 
     */
    void build(const vector<vector<bool>>& all_hashes, const MaskMatrix &sky_map ){

        hash_buckets_.clear();
        hash_buckets_.resize(num_masks_);
        doc_index_.clear();


        uint num_features = all_hashes.size();
        mask_size_ = sky_map.mask_size_;
        uint64_t total_postings =  all_hashes.size() * num_masks_ ;
        uint64_t postings_per_shard = ((total_postings + num_shards_ - 1) / num_shards_) * 103 / 100;
        doc_index_.resize(num_shards_);
        for(uint shard = 0; shard < num_shards_; shard++){
            doc_index_[shard].reserve(postings_per_shard);
        }

        // Generate Mask Hashes
        vector<vector<uint>> mask_hashes;
        mask_hashes.resize(num_masks_, vector<uint>(num_features));

        ProgressBar progress(num_masks_);
        std::atomic<uint64_t> completed_masks{0};
        cout<<"Generating Compressed Hashes "<<endl;
        #pragma omp parallel for collapse(2) schedule(static)
        for(uint mask_id = 0; mask_id < num_masks_; mask_id++){
            for(uint item_id = 0; item_id < num_features; item_id++){
                mask_hashes[mask_id][item_id] = sky_map.getCompressedHash(all_hashes[item_id], sky_map.masks_[mask_id]);
                if(item_id == num_features - 1)
                {
                    uint64_t done = completed_masks.fetch_add(1) + 1;
                    progress.update(done);
                }
            }
            
        }

        struct Entry {
            uint32_t key;
            uint32_t item_id;
        };

        // progress(num_masks_);
        completed_masks = 0;
        cout << "Building Inverted Index for Each Mask" << endl;
        #pragma omp parallel for num_threads(num_shards_)
        for(uint shard = 0; shard < num_shards_; shard++){
            for(uint mask_id = shard; mask_id < num_masks_; mask_id+=num_shards_){
                
                std:: vector<Entry> entries;
                double expected = static_cast<double>(num_features) / (1ull << sky_map.mask_size_);
                uint approx_entries_len = std::max(1u, static_cast<uint>(std::ceil(expected)));

                entries.reserve(10 * approx_entries_len);
                for(uint item_id = 0; item_id < num_features; item_id++){
                    
                    Entry x;
                    uint key = mask_hashes[mask_id][item_id];
                    x.key = key;
                    x.item_id = item_id;
                    entries.push_back(x);

                }

                // sort entries by key
                __gnu_parallel::sort( entries.begin(), entries.end(),
                    [](const Entry& a, const Entry& b) {
                        return a.key < b.key;
                    });

                for(int i = (int)entries.size() - 1 ; i >= 0; ){
                    uint key = entries[i].key;
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
                    hash_buckets_[mask_id].postings[key] = info;

                }
                uint64_t done = completed_masks.fetch_add(1) + 1;
                progress.update(done);
                                
            }
        }
    }

   /**
     * @brief Query Binary Hash is masked using the different masks of MaskMatrix sky_map to get compressed hashes for each
     * mask, the Inverted Index for that mask is then queried with the compressed hash value and the candidate neighbor 
     * item_ids are returned.
     *
     * @param query_hash vector<bool> The pre-computed binary hashes of the query vector.
     * @param sky_map MaskMatrix that holds all the binary masks.
     * @return set<uint> That contains item_ids of the possible candidate items.
     */

    inline std::tuple<vector<uint>, uint> get_matches(const vector<bool> &query_hash, const MaskMatrix &sky_map ){
        
        vector<uint> compressed_hashes;
        vector<uint> candidates;
        uint postings_traversed = 0;
        candidates.reserve(10000);
        
        compressed_hashes.resize(num_masks_);
        compressed_hashes = sky_map.getCompressedHash(query_hash);
        
        for( uint mask_id = 0; mask_id < sky_map.num_masks_; mask_id++ ){

            uint shard = mask_id % num_shards_; 
            uint key = compressed_hashes[mask_id];

            auto it = hash_buckets_[mask_id].postings.find(key);
            if(it == hash_buckets_[mask_id].postings.end()){
                continue;
            }

            uint start_index = it->second.start_index;
            uint item_count = it->second.item_count;

            for( uint i = start_index; i < start_index + item_count; i++  ){
                uint global_id = doc_index_[shard][i];
                if(!visited_[global_id]){
                    visited_[global_id] = 1;
                    candidates.push_back(global_id);
                }
                postings_traversed++;
                
            }
            
        }
        for(uint &item_id : candidates ){
                visited_[item_id] = 0; 
            }
        return { candidates, postings_traversed };

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
            bool q_h = query_hashes[h];
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
