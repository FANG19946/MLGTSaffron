#ifndef 7D4E5B12_A8F9_4C3D_9E71_2B6F8A4D1C90
#define 7D4E5B12_A8F9_4C3D_9E71_2B6F8A4D1C90



#include "headers.hpp"

struct HashNode
{
    vector<vector<vector<uint>>> postings;
};


/**
 * @brief A True Inverted Index that will store hashes and the global ids of the items that set them as follows
 * 
 * Hash Function range is 0 to R
 * We have L such hash functions
 * The inverted index will store an array of size R which will have the hash_vals and a pointer to an array of size L
 * Each array of size L will have a pointer to the an array of pool indices  
 * Each pool index will then have the list of the global ids that set that hash for that pool, hash function, hash val
 * How to access the elements: hash_buckets_[hash_val].postings[hash_func_num][pool_num][global_ids]
 * 
 */
class OrionIndex {
public:
    vector<HashNode> hash_buckets_; 
    uint hash_range_;
    uint num_hashes_;
    uint threshold_;

    /**
     * @brief Empty constructor for OrionIndex.
     */
    OrionIndex() : hash_range_(0), num_hashes_(0), threshold_(0) {}
    
    /**
     * @brief Construct a new Orion Index with specified parameters.
     * @param hash_range The range of the Hash Functions.
     * @param num_hashes The number of hashes (LSH functions) per item.
     * @param threshold The number of matching hashes required for a query match.
     */
    OrionIndex(uint hash_range, uint num_hashes, uint threshold) 
        : hash_range_(hash_range), num_hashes_(num_hashes), threshold_(threshold) {
        hash_buckets_.resize(hash_range);
        
    }

    /**
     * @brief Builds the inverted index from given item hashes.
     *  
     * @param all_hashes A 2D vector where all_hashes[i] contains the hashes for item i.
     * @param item_indices A 2D vector where item_indices[pool_id] contains the global ids of the items in that pool
     */
    void build(const vector<vector<uint>>& all_hashes, const vector<vector<uint>>& item_indices) {
        if (all_hashes.empty() || item_indices.empty()) return;
        
        num_hashes_ = all_hashes[0].size();
        num_pools = item_indices.size();
        
        hash_buckets_.resize(hash_range_);
        for(uint h_val = 0; h_val < hash_range_; h_val++ ){
            hash_buckets_[h_val].postings.resize(num_hashes_);
            for(uint h = 0; h < num_hashes_; h++){
                hash_buckets_[h_val].postings[h].resize(num_pools);
                
            }
        }

        
        
        uint N = item_indices.size();
        // for(uint h_val = 0; h_val < hash_range_; h_val++ ){
        //     for(uint h = 0; h < num_hashes_; h++){
        //         for(uint pool_id = 0; pool_id < num_pools; pool_id++){
        //             hash_buckets_[h_val].postings[h][pool_id]
        //         }
        //     }
        // }


        for(uint pool_id = 0; pool_id < num_pools; pool_id++ ){
            for(uint item : item_indices[i]){
                for(uint h = 0; h < num_hashes_; h++){
                    h_val = all_hashes[item][h];
                    hash_buckets_[h_val].postings[h][pool_id].push_back(item);
                }
            }
        }

        // 26-6-26





        for (uint h = 0; h < num_hashes_; ++h) {
            vector<pair<uint, uint>> hash_item_pairs;
            hash_item_pairs.reserve(N);
            for (uint i = 0; i < N; ++i) {
                uint global_idx = item_indices[i];
                hash_item_pairs.emplace_back(all_hashes[i][h], global_idx);
            }
            std::sort(hash_item_pairs.begin(), hash_item_pairs.end());

            for (uint i = 0; i < N; ) {
                uint h_val = hash_item_pairs[i].first;
                uint start_idx = i;
                while (i + 1 < N && hash_item_pairs[i + 1].first == h_val) i++;
                
                uint count = i - start_idx + 1;
                hash_buckets_[h].push_back({h_val, (uint)doc_index_.size(), count});
                for (uint j = start_idx; j <= i; ++j) {
                    doc_index_.push_back(hash_item_pairs[j].second);
                }
                i++;
            }
        }
    }

    /**
     * @brief Retrieves global item indices that match at least `threshold_` query hashes.
     * 
     * @param query_hashes The pre-computed hashes of the query vector.
     * @return vector<uint> A list of item indices that are candidates for similarity.
     */
    inline vector<uint> get_matches(const vector<uint> &query_hashes) const {
        if (num_hashes_ == 0 || doc_index_.empty()) return {};
        
        unordered_map<uint, uint> counts;
        for (uint h = 0; h < num_hashes_; ++h) {
            uint q_h = query_hashes[h];
            const auto& buckets = hash_buckets_[h];
            auto it = std::lower_bound(buckets.begin(), buckets.end(), q_h, 
                [](const HashBucket& b, uint val) { return b.hash_val < val; });
            
            if (it != buckets.end() && it->hash_val == q_h) {
                for (uint i = 0; i < it->num_items; ++i) {
                    counts[doc_index_[it->start_idx + i]]++;
                }
            }
        }

        vector<uint> matches;
        for (auto const& [item_idx, count] : counts) {
            if (count >= threshold_) {
                matches.push_back(item_idx);
            }
        }
        return matches;
    }
    
    /**
     * @brief Returns the number of hash functions the index expects.
     * @return uint Hash count.
     */
    inline uint num_hashes() const { return num_hashes_; }
};



#endif /* 7D4E5B12_A8F9_4C3D_9E71_2B6F8A4D1C90 */
