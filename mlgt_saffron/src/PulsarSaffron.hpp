#ifndef C8A7D1F2_5E3B_4A91_B6D4_91F8E7C2AB5D
#define C8A7D1F2_5E3B_4A91_B6D4_91F8E7C2AB5D


#include "headers.hpp"
#include "NovaMatrix.hpp"

/**
 * @brief The result vector has the test results for every test bundle. The idea is to create a vector of Test Bundles later to have a simpler debugging.
*/

struct TestBundle{
    vector<bool> result;
}


/**
 * @brief Generates the permutations needed for ids.
 * IMPORTANT
 * item_ids start from 0 but saffron signatures must start from 1 so even the first signature has item_id 0 -> signature_number 1.
 * We use random seeds sequentially starting from seed = 10.
 * @param num_permutations The number of permutations needed. For Saffron this is 3.
 * @return vector<vector<uint>> signature_map[item_id][j] gives the j-th signature_number for item_id.
 */

 inline vector<vector<uint>> getPermutationMap(uint num_permutations = 3){
    vector<uint> id_to_sign;
    vector<uint> permutation;
    vector<vector<uint>> permutation_map;
    permutation_map.resize(num_features_, vector<uint>(num_permutations));

    id_to_sign.resize(num_features_);
    
    for(uint i=0; i<num_features_; i++){
        id_to_sign[i]=i+1;
        permutation_map[i][0] = id_to_sign[i]; 
    }

    uint random_seed = 10;
    for(uint k=1; k < num_permutations; k++){
        permutation = id_to_sign;
        std::mt19937 rng(random_seed + k);   
        std::shuffle(permutation.begin(), permutation.end(), rng);
        for(uint i=0; i < num_features_; i++){
            permutation_map[i][k] = permutation[i];
        }
    }
    return permutation_map;
 }





/**
 * @brief Generates Signature for a block.
 * @param signature_number The signature_number of an item.
 * @return vector<bool> boolean signature of the block which includes the binary number concatenated with its complement.
 */
 inline vector<bool> getBlockSignature(uint signature_number){
    
    uint L = ceil(log2(num_features_)); // The L value in Saffron.
    vector<bool> block_signature(2*L);

    for(uint i=0; i<L; i++){
        block_signature[i] = (signature_number >> (L - 1 - i)) & 1;
        block_signature[i+L] = !block_signature[i];
    }

    return block_signature;
 }


/**
 * @brief Generates a robust 6L SAFFRON signature for singleton and doubleton recovery.
 * The boolean signature of the signature_number concatenated with its complement for all its permutations.
 * The signature consists of 3 blocks by default, each containing U1, and ~U1.
 * Block 1: Item signature which is item_id + 1 concatenated with its complement.
 * Block 2: Block Signature based on Permutation 1.
 * Block 3: Block Signature based on Permutation 2.
 * 
 * @param permutation_map Contains the permuations for the item index.
 * @return vector<bool> The boolean signature.
 */
 inline vector<bool> getSignature( const vector<uint>& permutation_map) {
    
    uint L = ceil(log2(num_features_)); // The L value in Saffron.
    uint num_permutations = permutation_map.size();
    
    vector<bool> signature;
    signature.reserve(num_permutations * (2 * L));  
    vector<bool> block_signature;

    for(auto permutation : permutation_map){
        block_signature = getBlockSignature(permutation);
        signature.insert(signature.end(),block_signature.begin(),block_signature.end());
    }

    return signature;
}

/**
 * @brief Small helper function that generate a 2D signature matrix which has the full signature for each item_id.
 
 * @param permutation_map The permutation_map which has signature_numbers for the item_ids.
 * @return vector<vector<bool>> signature_matrix[item_id] has full signature of item.
 */

 inline vector<vector<bool>> getSignatureMatrix(const vector<vector<uint>>& permutation_map) {
    vector<vector<bool>> signature_matrix(num_features_);
    for(uint item_id = 0; item_id < num_features_; item_id++){
        signature_matrix[item_id] = getSignature(permutation_map[item_id]); 
    }
    return signature_matrix;
 }


/**
 * @brief Generate the extended Pooling Matrix by row-wise replacement of item with its complete signature.
 * 
 * @param base_pooling_matrix The pooling matrix for the test bundles i.e. the left regular graph for test bundles.
 * @param signature_matrix vector<vector<bool>> signature_matrix[item_id] has full signature of item.
 * @return PoolingMatrix which is the Extended Pooling Matrix.
 */
 inline PoolingMatrix getExtendedPoolingMatrix(const PoolingMatrix &base_pooling_matrix, const vector<vector<bool>> &signature_matrix){
    PoolingMatrix extended_pooling_matrix;
    uint signature_length = num_permutations_ * 2 * L;
    extended_pooling_matrix.pools_to_items.resize(num_pools_ * signature_length);
    extended_pooling_matrix.items_to_pools.resize(num_features_);
    extended_pooling_matrix.num_features = num_features_;
    extended_pooling_matrix.num_pools = num_pools_ * signature_length;

    
    
    for( uint pool_id = 0; pool_id < base_pooling_matrix.pools_to_items.size(); pool_id++ ){
        uint extended_pool_id = pool_id * signature_length;
        
        for(uint i = 0; i < signature_length; i++ ){
            for(const uint &item_id : base_pooling_matrix.pools_to_items[pool_id]){
                bool bit = signature_matrix[item_id][i];
                if(bit){
                    extended_pooling_matrix.pools_to_items[extended_pool_id + i].push_back(item_id);
                    extended_pooling_matrix.items_to_pools[item_id].push_back(extended_pool_id + i);
                }

            }
        }
    }
    
    return extended_pooling_matrix;

 }

/**
 * @brief Decodes a single block of a signature that's the binary number concatenated with its complement.
 * IMPORTANT 
 * Remember that for the first block the binary number = item_id + 1.
 * @param block One block of the results vector it contains the binary signature number concatenated with its complement. 
 * Zeroton -> -1
 * Singleton -> permutation_number
 * Doubleton/ Multiton -> -2
 *  
 * @return Singleton returns permutation number, Zeroton returns -1, Multiton returns -2.
 */
inline int decodeBlock(const vector<bool>& block) {
    
    uint num_bits = block.size();
    int permutation_number = 0;
    uint hamming_weight = 0;

    for(bool &bit : block){
        hamming_weight += bit;
    }
    uint L = num_bits/2;
    // For hamming weight is half the num_bits that means its a singleton.
    if(hamming_weight == 0)
        permutation_number = -1;
    else if(hamming_weight == L){
        for(uint i=0; i<L; i++){
            if(block[i]){
                permutation_number += (1u << (L - 1 - i));
            }
        }
    }
    else{
        permutation_number=-2;
    }
    
    return permutation_number;
}


/**
 * @brief Decodes a signature from a measurement vector.
 * 
 * @param measurement A boolean vector of measurement bits (6L Generally).
 * @param permutation_map The permutation_map which has signature_numbers for the item_ids.
 * Zeroton -> -1
 * Singleton -> item_id
 * Doubleton/ Multiton -> -2
 * Decoding Failure -> -3 ( Permutation verification failure )
 * @return item_id for successful decoding and negative codes for failures.
 */
inline int decodeSignature(
    const vector<bool>& measurement, const vector<vector<uint>> &permutation_map
    
) {
    uint block_len = signature_length_ / num_permutations_;
    uint item_id;

    // Decoding First Block.
    vector<bool> block(
    measurement.begin(),
    measurement.begin() + block_len
    );

    int decoded = decodeBlock(block);
    // Zeroton or Multiton.
    if(decoded < 0){
        return decoded;
    }
    else {
        // verify with other blocks
        uint permutation_number = decoded;
        item_id = permutation_number - 1;
        
        for(uint i = 1; i< num_permutations_; i++){
            block = vector<bool>( measurement.begin() + i * block_len, measurement.begin() + (i+1) * block_len );
            int next_permutation = decodeBlock(block);
            if(next_permutation < 0){
                return -3;
            }
            if(permutation_map[item_id][i] != next_permutation ){
                return -3;
            }
        }
        return item_id;
    }

}

/**
 * @brief Peel a Signature of an identified defective from a measurement.
 * 
 * @param measurement A boolean vector of measurement bits (6L Generally).
 * @param signature_matrix vector<vector<bool>> signature_matrix[item_id] has full signature of item.
 * @param defective_item_id item_id of the identified defective in the bundle.
 * 
 * Resolvable Doubleton -> item_id
 * Decoding Failure -> -3 ( Permutation verification failure )
 * @return vector<bool> result The peeled signature of the measurement vector.
 */
inline vector<bool> peelSignature(const vector<bool> &measurement, const vector<vector<bool>> &signature_matrix, uint defective_item_id){
    
    vector<bool> result(signature_length_);
    vector<bool> &defective_signature = signature_matrix[defective_item_id];
    for(uint i=0; i< num_permutations_; i++){
        for(uint j=0; j< L_; j++){

            uint bit_id = i*2*L+j;
            uint bit_complement_id = bit_id + L_;

            
            // 1 OR Unknown != 0
            assert(!(measurement[bit_id] == 0 && defective_signature[bit_id] == 1));
            
            if(measurement[bit_id] == 1){
                if(defective_signature[bit_id] == 1){
                    if(measurement[bit_complement_id] == 1){
                        result[bit_id] = 0;
                    }
                    else{
                        result[bit_id] = 1;
                    }
                }
                else{
                    result[bit_id] = 1;
                }
            }
            else{
                result[bit_id] = 0;
            }

            result[bit_complement_id] = !result[bit_id];

        }
    }
    return result;
}



/**
 * @brief Base class for the Sparse All-Fast Fourier Transform (SAFFRON) recovery algorithm.
 * 
 * Provides core mechanisms for pooling, signature generation, and the iterative peeling 
 * algorithm used to recover sparse components (singletons and doubletons) from 
 * XOR-sum binary residuals across multiple pools.
 */
class Saffron {
protected:
    PoolingMatrix pools_; // The pooling matrix defining the item-to-pool and pool-to-item mappings.
    uint num_features_; // Total number of features/items (n).
    uint sparsity_; // Expected sparsity level (k).
    uint num_pools_; // Total number of pools (m).
    uint signature_length_; // Length of the signatures.
    int debug_ = 0; // Debug level (0 for none, higher values for more verbose output)
    uint num_permutations_ = 3; // The number of permuations used by Saffron.

public:
    /**
     * @brief Initializes the Saffron algorithm setup.
     * 
     * @param num_features Total number of features (n).
     * @param sparsity Expected sparsity level (k).
     * @param debug Debug level.
     */
    Saffron(uint num_features, uint sparsity, int debug = 0, uint num_permuations = 3) :
        num_features_(num_features),
        sparsity_(sparsity),
        debug_(debug),
        num_permuations_(num_permuations)
    {
        uint L = ceil(log2(num_features));
        signature_length_ = 3 * (2 * L + 1);
        pools_ = computePools(num_features, sparsity, debug);
        num_pools_ = pools_.num_pools;
    }

    /**
     * @brief Returns the number of features (n).
     * @return uint 
     */
    inline uint num_features() const { return num_features_; }

    /**
     * @brief Returns the total number of features (n).
     * @return uint 
     */
    inline uint size() const { return num_features_; }

    /**
     * @brief Returns the expected sparsity level (k).
     * @return uint 
     */
    inline uint sparsity() const { return sparsity_; }

    /**
     * @brief Returns the number of pools.
     * @return uint 
     */
    inline uint num_pools() const { return num_pools_; }

    /**
     * @brief Returns the length of the signatures.
     * @return uint 
     */
    inline uint signature_length() const { return signature_length_; }

    ~Saffron() = default;

    /**
     * @brief Executes the peeling algorithm to recover identified items from residuals.
     * 
     * @param residuals A 2D vector of booleans (num_pools x signature_bits).
     * @param debug Debug level.
     * @return set<uint> A set of identified item indices.
     */
    inline set<uint> peelingAlgorithm(vector<vector<bool>> residuals, int debug = 0) {
        set<uint> identified;
        queue<Candidate> candidates;
        assert(residuals.size() == pools_.num_pools && "Residuals size mismatch");

        auto check_pool = [&](uint p_idx) {
            vector<uint> decoded = decodeSignature(residuals[p_idx], signature_length_);
            for (uint it : decoded) {
                if (it < num_features_) {
                    candidates.push({p_idx, it});
                }
            }
        };

        for (uint pool_idx = 0; pool_idx < pools_.num_pools; ++pool_idx) {
            check_pool(pool_idx);
        }

        if (debug_ > 0 || debug > 0) {
            cout << "Initialized peeling with " << candidates.size() 
                << " candidates." << endl;
        }

        while (!candidates.empty()) {
            Candidate cand = candidates.front();
            candidates.pop();

            if (identified.count(cand.item_idx)) continue;
            
            identified.insert(cand.item_idx);
            
            vector<bool> sig = getSignature(cand.item_idx, signature_length_);
            for (uint p_idx : pools_.items_to_pools[cand.item_idx]) {
                for (uint b = 0; b < signature_length_; ++b) {
                    if (sig[b]) residuals[p_idx][b].flip();
                }
                check_pool(p_idx);
            }
        }
        return identified;
    }
};


#endif // C8A7D1F2_5E3B_4A91_B6D4_91F8E7C2AB5D

