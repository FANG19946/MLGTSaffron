#ifndef H_7D4E5B12_A8F9_4C3D_9E71_2B6F8A4D1C90
#define H_7D4E5B12_A8F9_4C3D_9E71_2B6F8A4D1C90



#include "headers.hpp"
#include <algorithm>
#include <execution>

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
    vector<uint> doc_index_; 

    /**
     * @brief Empty constructor for OrionIndex.
     */
    OrionIndex() : hash_range_(0), num_hashes_(0), threshold_(0), num_pools_(0) {}
    
    /**
     * @brief Construct a new Orion Index with specified parameters.
     * @param num_pools_ The number of pools.
     * @param num_hashes The number of hashes (LSH functions) per item.
     * @param threshold The number of matching hashes required for a query match.
     */
    OrionIndex(uint num_pools, uint num_hashes, uint threshold) 
        : num_pools_(num_pools), num_hashes_(num_hashes), threshold_(threshold) {
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

        doc_index_.reserve(POOLS_PER_ITEM * all_hashes.size() * num_hashes_ * L * 3);

        struct Entry {
            uint32_t key;
            uint32_t item_id;
        };

        for(uint pool_id = 0; pool_id < num_pools_; pool_id++ ){
            
            std::vector<Entry> entries;
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
            std::sort(std::execution::par, entries.begin(), entries.end(),
                [](const Entry& a, const Entry& b) {
                    return a.key < b.key;
                });
            
            for(int i = entries.size() - 1 ; i >= 0; ){
                uint key = entries[i].key;
                // int j =  i - 1;
                uint start_index = doc_index_.size();
                uint item_count = 1;
                PostingRange info;
                info.start_index = start_index;
                doc_index_.push_back(entries[i].item_id);
                i--;
                while(i >= 0 && key == entries[i].key){
                    item_count++;
                    doc_index_.push_back(entries[i].item_id);
                    i--;
                    // j = i - 1;
                    
                }
              

                info.item_count = item_count;
                hash_buckets_[pool_id].postings[key] = info;
                


            }
        }
        all_hashes.clear();
        all_hashes.shrink_to_fit();
        size_t expected = 0;
        for (const auto &pool : item_indices)
            expected += pool.size() * num_hashes_;

        cout << "Expected postings = " << expected << endl;
        cout << "Actual doc_index size = " << doc_index_.size() << endl;
        size_t total_counts = 0;

        for (const auto& bucket : hash_buckets_) {
            for (const auto& [key, range] : bucket.postings) {
                total_counts += range.item_count;

                assert(range.start_index + range.item_count <= doc_index_.size());
            }
        }

        cout << "Total posting counts = " << total_counts << endl;

        
        

        
        
        // for(uint pool_id = 0; pool_id < num_pools_; pool_id++ ){
        //     for(uint item : item_indices[pool_id]){
        //         for(uint h = 0; h < num_hashes_; h++){
        //             uint h_val = all_hashes[item][h];
        //             uint key = getPoolHashKey(h, pool_id);
        //             hash_buckets_[h_val].postings[key].push_back(item);
        //         }
        //     }
        //     // For Debugging
        //     if(pool_id%100 == 0){
        //         cout<<pool_id<<" Pools Procssed"<<endl;
        //         cout<<std::flush;
        //     }
        // }

        // double gb = memoryUsage() / (1024.0 * 1024.0 * 1024.0);

        // std::cout << "Approximate index size = "
        //         << gb
        //         << " GB\n";
        
    }


    // inline size_t memoryUsage() const {
    //     size_t bytes = 0;

    //     // OrionIndex object itself
    //     bytes += sizeof(*this);

    //     // hash_buckets_ vector allocation
    //     bytes += hash_buckets_.capacity() * sizeof(HashNode);

    //     for (const auto& bucket : hash_buckets_) {

    //         // unordered_map bucket array
    //         bytes += bucket.postings.bucket_count() * sizeof(void*);

    //         // each hashmap node
    //         bytes += bucket.postings.size() *
    //                 sizeof(decltype(bucket.postings)::value_type);

    //         // posting lists
    //         for (const auto& [key, posting] : bucket.postings) {
    //             bytes += posting.capacity() * sizeof(uint);
    //         }
    //     }

    //     return bytes;
    // }

    /**
     * @brief It returns the result of a test as 0 (Negative) or 1 (Positive) and the defective item identified.
     * IMPORTANT
     * Please note that if the test result is negaative the uint value is set by default to 0.
     * @param query_hashes The pre-computed hashes of the query vector.
     * @param pool_index Index of the pool being evaluated.
     * @return pair<bool, uint> Test Result and if positive also has the global_id of the positive item.
     */
    inline pair<bool, uint> get_matches(const vector<uint> &query_hashes, uint pool_index) const {
        if (num_hashes_ == 0 ) return {false, 0};
        
                    
        unordered_map<uint, uint> counts;
        uint hash_misses = 0; 

        for(uint h = 0; h < num_hashes_ ; h++){
            uint q_h = query_hashes[h];
            uint key = getPoolHashKey(h, q_h);

            // Checking if query hash can match threshold
            if(hash_misses > num_hashes_ - threshold_){
                return {false, 0};
            }
            
            // Skip if Query Hash doesn't exist in Index
            auto it = hash_buckets_[pool_index].postings.find(key);
            if(it == hash_buckets_[pool_index].postings.end()){
                hash_misses++;
                continue;
            }

            uint start_index = it->second.start_index;
            uint item_count = it->second.item_count;
            // it->second is just hash_buckets_[q_h].postings[key] 
            for( uint i = start_index; i < start_index + item_count; i++  ){
                uint global_id = doc_index_[i];
                if(++counts[global_id] >= threshold_){
                    return {true, global_id};
                }
            }              
        }
        return {false, 0};        
    }
    
    /**
     * @brief Returns the number of hash functions the index expects.
     * @return uint Hash count.
     */
    // inline uint num_hashes() const { return num_hashes_; }
};



#endif /* H_7D4E5B12_A8F9_4C3D_9E71_2B6F8A4D1C90 */
