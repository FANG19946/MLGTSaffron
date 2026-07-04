#ifndef C8A7D1F2_5E3B_4A91_B6D4_91F8E7C2AB5D
#define C8A7D1F2_5E3B_4A91_B6D4_91F8E7C2AB5D


#include "headers.hpp"
#include "PoolingMatrix.hpp"


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
    signature_map.resize(num_features_, vector<uint>(num_permutations));

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
 * @brief Generate the extended Pooling Matrix by row-wise replacement of item with it's complete signature.
 * 
 * @param base_pooling_matrix The pooling matrix for the test bundles i.e. the left regular graph for test bundles.
 * @param permutation_map The permutation_map which has signature_numbers for the item_ids.
 * @return PoolingMatrix which is the Extended Pooling Matrix.
 */
 inline PoolingMatrix getExtendedPoolingMatrix(const PoolingMatrix &base_pooling_matrix, const vector<vector<uint>> &permutation_map){
    PoolingMatrix extended_pooling_matrix;
    vector<vector<bool>> signature_matrix = getSignatureMatrix(permutation_map);
    vector<vector<uint>> extended_pools_to_items;




 }

/**
 * @brief Decodes a single block of a signature.
 * 
 * @param measurement A slice of the measurement vector for one block.
 * @return optional<uint> The decoded item index, or nullopt if 0 or multi-pool.
 */
inline optional<uint> decodeBlock(const vector<bool>& measurement) {
    if (!measurement[0]) return nullopt; // Parity 0 -> even/zero items
    uint num_bits = (measurement.size() - 1) / 2;
    uint index = 0;
    for (uint bit_idx = 0; bit_idx < num_bits; ++bit_idx) {
        bool b1 = measurement[1 + bit_idx];
        bool b2 = measurement[1 + num_bits + bit_idx];
        if (b1 == b2) return nullopt; // Collision or empty
        if (b1) index |= (1 << bit_idx);
    }
    return index;
}


/**
 * @brief Decodes a signature from a measurement vector, supporting doubleton resolution.
 * 
 * @param measurement A boolean vector of measurement bits (6L).
 * @param signature_length Length of the signature.
 * @return vector<uint> List of identified item indices (0, 1, or 2 items).
 */
inline vector<uint> decodeSignature(
    const vector<bool>& measurement,
    uint signature_length
) {
    uint block_len = signature_length / 3;
    uint num_bits = (block_len - 1) / 2;
    
    // 1. Try singleton decoding from Block 1
    vector<bool> block1(measurement.begin(), measurement.begin() + block_len);
    optional<uint> s = decodeBlock(block1);
    if (s.has_value()) {
        uint val = s.value();
        vector<bool> expected = getSignature(val, signature_length);
        bool match = true;
        for (uint i = 0; i < signature_length; ++i) {
            if (expected[i] != measurement[i]) {
                match = false;
                break;
            }
        }
        if (match) return { val };
    }
    
    // 2. Try doubleton recovery if block 1 parity is 0
    if (!block1[0]) {
        uint S = 0;
        bool non_zero = false;
        for (uint i = 0; i < num_bits; ++i) {
            if (block1[1 + i]) { S |= (1 << i); non_zero = true; }
        }
        if (!non_zero) return {}; // Truly empty

        // Look for a singleton in other blocks
        for (uint b = 1; b < 3; ++b) {
            vector<bool> block_m(measurement.begin() + b * block_len, measurement.begin() + (b + 1) * block_len);
            optional<uint> i1 = decodeBlock(block_m);
            if (i1.has_value()) {
                uint val1 = i1.value();
                uint val2 = S ^ val1;
                
                // Verify doubleton
                vector<bool> s1 = getSignature(val1, signature_length);
                vector<bool> s2 = getSignature(val2, signature_length);
                bool match = true;
                for (uint i = 0; i < signature_length; ++i) {
                    if ((s1[i] ^ s2[i]) != measurement[i]) {
                        match = false;
                        break;
                    }
                }
                if (match) return { val1, val2 };
            }
        }
    }
    
    return {};
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

public:
    /**
     * @brief Initializes the Saffron algorithm setup.
     * 
     * @param num_features Total number of features (n).
     * @param sparsity Expected sparsity level (k).
     * @param debug Debug level.
     */
    Saffron(uint num_features, uint sparsity, int debug = 0) :
        num_features_(num_features),
        sparsity_(sparsity),
        debug_(debug)
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

