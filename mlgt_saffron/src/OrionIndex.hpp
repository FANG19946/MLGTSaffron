#ifndef H_7D4E5B12_A8F9_4C3D_9E71_2B6F8A4D1C90
#define H_7D4E5B12_A8F9_4C3D_9E71_2B6F8A4D1C90



#include "headers.hpp"
#include <algorithm>
#include <execution>
#include <fstream>
#include <stdexcept>
#include <parallel/algorithm>
#include "ProgressBar.hpp"

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
    uint num_shards_;
    vector<vector<uint>> doc_index_; 

    /**
     * @brief Empty constructor for OrionIndex.
     */
    OrionIndex() : hash_range_(0), num_hashes_(0), threshold_(0), num_pools_(0), num_shards_(8) {}
    
    /**
     * @brief Construct a new Orion Index with specified parameters.
     * @param num_pools_ The number of pools.
     * @param num_hashes The number of hashes (LSH functions) per item.
     * @param threshold The number of matching hashes required for a query match.
     */
    OrionIndex(uint num_pools, uint num_hashes, uint threshold) 
        : num_pools_(num_pools), num_hashes_(num_hashes), threshold_(threshold),  num_shards_(8) {
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

        uint64_t total_postings = POOLS_PER_ITEM * all_hashes.size() * num_hashes_ * L * 3;
        uint64_t postings_per_shard = ((total_postings + num_shards_ - 1) / num_shards_) * 102 / 100;
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
        for(uint pool_id = 0; pool_id < num_pools_; pool_id++ ){
            
            std::vector<Entry> entries;
            entries.reserve(39000000);
            for(uint item : item_indices[pool_id]){
                
                for(uint h = 0;  h < num_hashes_; h++){
                    Entry x;
                    uint h_val = all_hashes[item][h];
                    uint key = getPoolHashKey(h, h_val);
                    x.key = key;
                    x.item_id = item;
                    // if(item == 341576 && h == 0)
                    // {
                    //     cout << "INSERTING ITEM 341576"
                    //         << " hash=" << h_val
                    //         << " pool=" << pool_id
                    //         << endl;
                    // }
                    entries.push_back(x);

                }

                

            }

            // if(pool_id == 1801)
            // {
            //     uint target_key = getPoolHashKey(0, 200021);

            //     bool found = false;

            //     for(auto &e : entries)
            //     {
            //         if(e.key == target_key && e.item_id == 341576)
            //         {
            //             found = true;
            //             break;
            //         }
            //     }

            //     cout << "POOL 1801 BEFORE SORT: "
            //         << (found ? "FOUND" : "MISSING")
            //         << endl;
            // }

            // sort entries by hash_val and then among same hash_vals by key
            __gnu_parallel::sort( entries.begin(), entries.end(),
                [](const Entry& a, const Entry& b) {
                    return a.key < b.key;
                });

            // if(pool_id == 1801)
            // {
            //     uint target_key = getPoolHashKey(0,200021);

            //     for(auto &e : entries)
            //     {
            //         if(e.key == target_key)
            //         {
            //             cout<<"POOL 1801 SORTED ENTRY "
            //                 <<"item="<<e.item_id
            //                 <<" key="<<e.key
            //                 <<endl;
            //         }
            //     }
            // }
            
            for(int i = (int)entries.size() - 1 ; i >= 0; ){
                uint key = entries[i].key;
                // int j =  i - 1;
                uint shard = pool_id % num_shards_;
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
                    // j = i - 1;
                    
                }
              

                info.item_count = item_count;
                hash_buckets_[pool_id].postings[key] = info;
                // if(pool_id == 1801 && key == getPoolHashKey(0,200021))
                // {
                //     cout<<"CREATED POSTING\n";
                //     cout<<"start="<<info.start_index<<"\n";
                //     cout<<"count="<<info.item_count<<"\n";

                //     for(uint j=info.start_index;
                //         j<info.start_index+info.item_count;
                //         j++)
                //     {
                //         cout<<"doc_index["<<j<<"]="
                //             <<doc_index_[shard][j]<<endl;
                //     }
                // }

            }
            progress.update(pool_id + 1);

        }
        // all_hashes.clear();
        // all_hashes.shrink_to_fit();
        // for(uint pool = 0; pool < item_indices.size(); pool++)
        // {
        //     for(uint item : item_indices[pool])
        //     {
        //         if(item == 341576)
        //         {
        //             cout << "BUILD SEES ITEM 341576\n";
        //             cout << "pool = " << pool << endl;
        //             cout << "hashes: ";

        //             for(auto x : all_hashes[item])
        //                 cout << x << " ";

        //             cout << endl;
        //         }
        //     }
        // }
        

        
        

        
        
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


    void debug_lookup(uint pool, uint key) {
        auto it = hash_buckets_[pool].postings.find(key);
        if (it == hash_buckets_[pool].postings.end())
            cout << "NOT FOUND\n";
        else
            cout << it->second.start_index << " "
                << it->second.item_count << endl;
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
        uint shard = pool_index % num_shards_; 

        for(uint h = 0; h < num_hashes_ ; h++){
            uint q_h = query_hashes[h];
            uint key = getPoolHashKey(h, q_h);

            // Checking if query hash can match threshold
            // if(hash_misses > num_hashes_ - threshold_){
            //     return {false, 0};
            // }
            
            // Skip if Query Hash doesn't exist in Index
            auto it = hash_buckets_[pool_index].postings.find(key);
            if(it == hash_buckets_[pool_index].postings.end()){
                hash_misses++;
                continue;
            }

            uint start_index = it->second.start_index;
            uint item_count = it->second.item_count;
            // it->second is just hash_buckets_[pool_index].postings[key] 

            // if(pool_index == 1801 &&
            // h == 0 &&
            // q_h == 200021)
            // {
            //     cout<<"QUERY FOUND POSTING\n";
            //     cout<<"start="<<start_index<<"\n";
            //     cout<<"count="<<item_count<<"\n";

            //     for(uint j=start_index;
            //         j<start_index+item_count;
            //         j++)
            //     {
            //         cout<<"READ doc_index["<<j<<"]="
            //             <<doc_index_[shard][j]<<endl;
            //     }
            // }

            for( uint i = start_index; i < start_index + item_count; i++  ){
                uint global_id = doc_index_[shard][i];
                if(++counts[global_id] >= threshold_){
                    // cout<<"Positive item_id : "<<global_id<<endl<<"Pool Index : "<<pool_index<<endl;
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
