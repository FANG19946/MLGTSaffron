#ifndef D92A6E5B_1C48_4F37_A9E1_72B4D8C5F0A6
#define D92A6E5B_1C48_4F37_A9E1_72B4D8C5F0A6

#include "headers.hpp"
#include "ProgressBar.hpp"
#include <atomic>

class MaskMatrix{
public:
    vector<vector<uint>> masks_;
    uint num_features_;
    uint num_hashes_;
    uint mask_size_;
    uint num_masks_;

    MaskMatrix()
    : num_features_(0),
      num_hashes_(0),
      mask_size_(0),
      num_masks_(0)
{}


    MaskMatrix(uint num_features, uint num_hashes, uint num_masks, uint mask_size)
    :
        num_features_(num_features),
        num_hashes_(num_hashes),
        num_masks_(num_masks),
        mask_size_(mask_size)
    {

        // assert(num_features_ > 0);
        // uint L = ceil(log2(num_features));
        // if(C_mask_bits < 0){
        //     int x = -L;
        //     assert(C_mask_bits > x);
        // }
        // L = L + C_mask_bits;
        // mask_size_ = L;

        

        vector<vector<bool>> all_masks;
        all_masks.resize(num_masks_, vector<bool> (num_hashes_, false ));

        assert(mask_size_ <= num_hashes_);
        assert(num_hashes_ > 0);


        ProgressBar progress(num_masks_);
        std::atomic<uint64_t> completed_masks{0};
        #pragma omp parallel for
        for(uint mask_id = 0; mask_id < num_masks_; mask_id++){

            std::mt19937 gen(42 + mask_id);
            std::uniform_int_distribution<int> dist(0, num_hashes_ - 1);

            for(uint active_position = 0; active_position < mask_size_; ){
                uint pos = dist(gen);
                if(!all_masks[mask_id][pos]){
                    all_masks[mask_id][pos] = true;
                    active_position++;
                }
            }
            uint64_t done = completed_masks.fetch_add(1) + 1;
            progress.update(done);
        }

        masks_.resize(num_masks_);
        for(uint mask_id = 0; mask_id < num_masks_; mask_id++){
            masks_[mask_id].reserve(mask_size_);
            for(uint h = 0; h < num_hashes_; h++ ){
                if(all_masks[mask_id][h]){
                    masks_[mask_id].push_back(h);
                }
            }
            assert(masks_[mask_id].size() == mask_size_);

        }

    }

    inline uint getCompressedHash(const vector<bool> &binary_hash, const vector<uint> &mask) const {

        uint hash_val = 0;        
        for(uint mask_bit = 0; mask_bit < mask.size(); mask_bit++){
            uint pos = mask[mask_bit];
            hash_val = (hash_val << 1) | static_cast<uint>(binary_hash[pos]);
        }
        return hash_val;
    }

    inline vector<uint> getCompressedHash(const vector<bool> &binary_hash) const {
        vector<uint> compressed_hashes(num_masks_);
        for(uint mask_id = 0; mask_id < num_masks_; mask_id++){
            compressed_hashes[mask_id] = getCompressedHash(binary_hash, masks_[mask_id]);        
        }
        return compressed_hashes;
    }


};


#endif /* D92A6E5B_1C48_4F37_A9E1_72B4D8C5F0A6 */