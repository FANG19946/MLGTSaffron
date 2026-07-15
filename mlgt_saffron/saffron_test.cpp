#pragma once
#include <cassert>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <unordered_set>
#include <vector>
#include <set>

using namespace std;

using uint = unsigned int;

// -----------------------------------------------------------------------------
// Replace these with the same values from your project.
// -----------------------------------------------------------------------------

constexpr uint C_epsilon = 3;
constexpr uint POOLS_PER_ITEM = 3;

// -----------------------------------------------------------------------------

/**
 * @brief Represents the mapping between items and pools.
 */
struct PoolingMatrix
{
    vector<vector<uint>> pools_to_items;
    vector<vector<uint>> items_to_pools;
    uint num_features;
    uint num_pools;
};

/**
 * @brief Randomized pooling matrix generation.
 *
 * Standard Left Regular Graph.
 *
 * @param num_features Total number of features (n).
 * @param sparsity Expected sparsity level (k).
 * @param debug Debug verbosity.
 * @return PoolingMatrix
 */
inline PoolingMatrix computePools(
    uint num_features,
    uint sparsity,
    int debug = 0)
{
    uint num_pools = sparsity * C_epsilon;
    uint pools_per_item = POOLS_PER_ITEM;

    PoolingMatrix pooling_matrix;

    pooling_matrix.num_features = num_features;
    pooling_matrix.num_pools = num_pools;

    pooling_matrix.pools_to_items.resize(num_pools);
    pooling_matrix.items_to_pools.resize(num_features);

    std::mt19937 gen(42);
    std::uniform_int_distribution<uint> dis(0, num_pools - 1);

    for (uint item_idx = 0; item_idx < num_features; item_idx++)
    {

        unordered_set<uint> chosen_pools;

        while (chosen_pools.size() < pools_per_item)
        {
            chosen_pools.insert(dis(gen));
        }

        for (uint pool_idx : chosen_pools)
        {
            pooling_matrix.pools_to_items[pool_idx].push_back(item_idx);
            pooling_matrix.items_to_pools[item_idx].push_back(pool_idx);
        }
    }

    if (debug > 0)
    {
        cout << "[Compute Pools]\n";
        cout << "num_features  : " << num_features << '\n';
        cout << "sparsity      : " << sparsity << '\n';
        cout << "num_pools     : " << num_pools << '\n';
        cout << "pools/item    : " << pools_per_item << '\n';
    }

    return pooling_matrix;
}

/**
 * @brief Generates the permutations needed for ids.
 * IMPORTANT
 * item_ids start from 0 but saffron signatures must start from 1 so even the first
 * signature has item_id 0 -> signature_number 1.
 * We use random seeds sequentially starting from seed = 10.
 *
 * @param num_features Number of features/items.
 * @param num_permutations The number of permutations needed. For Saffron this is 3.
 * @return vector<vector<uint>> permutation_map[item_id][j] gives the j-th
 * signature_number for item_id.
 */
inline vector<vector<uint>> getPermutationMap(
    uint num_features,
    uint num_permutations = 3)
{
    vector<uint> id_to_sign;
    vector<uint> permutation;
    vector<vector<uint>> permutation_map;

    permutation_map.resize(num_features, vector<uint>(num_permutations));

    id_to_sign.resize(num_features);

    for (uint i = 0; i < num_features; i++)
    {
        id_to_sign[i] = i + 1;
        permutation_map[i][0] = id_to_sign[i];
    }

    uint random_seed = 10;

    for (uint k = 1; k < num_permutations; k++)
    {
        permutation = id_to_sign;

        std::mt19937 rng(random_seed + k);
        std::shuffle(permutation.begin(), permutation.end(), rng);

        for (uint i = 0; i < num_features; i++)
        {
            permutation_map[i][k] = permutation[i];
        }
    }

    return permutation_map;
}

/**
 * @brief Generates Signature for a block.
 *
 * @param signature_number The signature_number of an item.
 * @param num_features Total number of features/items (n).
 * @return vector<bool> Boolean signature of the block which includes the binary
 * number concatenated with its complement.
 */
inline vector<bool> getBlockSignature(
    uint signature_number,
    uint num_features)
{
    uint L = ceil(log2(num_features)); // The L value in Saffron.

    vector<bool> block_signature(2 * L);

    for (uint i = 0; i < L; i++)
    {
        block_signature[i] = (signature_number >> (L - 1 - i)) & 1;
        block_signature[i + L] = !block_signature[i];
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
 * @param permutation_map Contains the permutations for the item index.
 * @param num_features Total number of features/items (n).
 * @return vector<bool> The boolean signature.
 */
inline vector<bool> getSignature(
    const vector<uint> &permutation_map,
    uint num_features)
{
    uint L = ceil(log2(num_features)); // The L value in Saffron.
    uint num_permutations = permutation_map.size();

    vector<bool> signature;
    signature.reserve(num_permutations * (2 * L));

    vector<bool> block_signature;

    for (auto permutation : permutation_map)
    {
        block_signature = getBlockSignature(permutation, num_features);
        signature.insert(
            signature.end(),
            block_signature.begin(),
            block_signature.end());
    }

    return signature;
}

/**
 * @brief Small helper function that generates a 2D signature matrix which has
 * the full signature for each item_id.
 *
 * @param permutation_map The permutation_map which has signature_numbers for the item_ids.
 * @param num_features Total number of features/items (n).
 * @return vector<vector<bool>> signature_matrix[item_id] has full signature of item.
 */
inline vector<vector<bool>> getSignatureMatrix(
    const vector<vector<uint>> &permutation_map,
    uint num_features)
{
    vector<vector<bool>> signature_matrix(num_features);

    for (uint item_id = 0; item_id < num_features; item_id++)
    {
        signature_matrix[item_id] = getSignature(
            permutation_map[item_id],
            num_features);
    }

    return signature_matrix;
}

/**
 * @brief Generate the extended Pooling Matrix by row-wise replacement of item with its complete signature.
 *
 * @param base_pooling_matrix The pooling matrix for the test bundles i.e. the left regular graph for test bundles.
 * @param signature_matrix signature_matrix[item_id] has full signature of item.
 * @param num_features Total number of features/items (n).
 * @param num_permutations Number of permutations used to generate the signatures.
 * @return PoolingMatrix The Extended Pooling Matrix.
 */
inline PoolingMatrix getExtendedPoolingMatrix(
    const PoolingMatrix &base_pooling_matrix,
    const vector<vector<bool>> &signature_matrix,
    uint num_features,
    uint num_permutations)
{
    uint L = ceil(log2(num_features));
    uint signature_length = num_permutations * 2 * L;

    PoolingMatrix extended_pooling_matrix;

    extended_pooling_matrix.pools_to_items.resize(
        base_pooling_matrix.num_pools * signature_length);

    extended_pooling_matrix.items_to_pools.resize(num_features);

    extended_pooling_matrix.num_features = num_features;
    extended_pooling_matrix.num_pools =
        base_pooling_matrix.num_pools * signature_length;

    for (uint pool_id = 0;
         pool_id < base_pooling_matrix.pools_to_items.size();
         pool_id++)
    {
        uint extended_pool_id = pool_id * signature_length;

        for (uint i = 0; i < signature_length; i++)
        {

            for (const uint &item_id :
                 base_pooling_matrix.pools_to_items[pool_id])
            {
                bool bit = signature_matrix[item_id][i];

                if (bit)
                {
                    extended_pooling_matrix
                        .pools_to_items[extended_pool_id + i]
                        .push_back(item_id);

                    extended_pooling_matrix
                        .items_to_pools[item_id]
                        .push_back(extended_pool_id + i);
                }
            }
        }
    }

    return extended_pooling_matrix;
}
/**
 * @brief Decodes a single block of a signature that's the binary number
 * concatenated with its complement.
 *
 * Zeroton    -> -1
 * Singleton  -> permutation_number
 * Multiton   -> -2
 */
inline int decodeBlock(const vector<bool> &block)
{
    uint num_bits = block.size();
    uint L = num_bits / 2;

    uint hamming_weight = 0;
    for (bool bit : block)
        hamming_weight += bit;

    if (hamming_weight == 0)
        return -1;

    if (hamming_weight != L)
        return -2;

    int permutation_number = 0;

    for (uint i = 0; i < L; i++)
    {
        if (block[i])
            permutation_number += (1u << (L - 1 - i));
    }

    return permutation_number;
}
/**
 * @brief Decodes a complete SAFFRON signature.
 *
 * Zeroton                -> -1
 * Singleton              -> item_id
 * Doubleton / Multiton   -> -2
 * Verification Failure   -> -3
 */
inline int decodeSignature(
    const vector<bool> &measurement,
    const vector<vector<uint>> &permutation_map)
{
    uint num_permutations = permutation_map[0].size();
    uint block_len = measurement.size() / num_permutations;

    // ---------- Decode first block ----------

    vector<bool> block(
        measurement.begin(),
        measurement.begin() + block_len);

    int decoded = decodeBlock(block);

    // Zeroton or Multiton
    if (decoded < 0)
        return decoded;

    uint permutation_number = decoded;
    uint item_id = permutation_number - 1;

    // Invalid permutation number
    if (item_id >= permutation_map.size())
        return -3;

    // ---------- Verify remaining blocks ----------

    for (uint i = 1; i < num_permutations; i++)
    {
        block.assign(
            measurement.begin() + i * block_len,
            measurement.begin() + (i + 1) * block_len);

        int next_permutation = decodeBlock(block);

        if (next_permutation < 0)
            return -3;

        if (permutation_map[item_id][i] !=
            static_cast<uint>(next_permutation))
            return -3;
    }

    return item_id;
}

/**
 * @brief Peel a known defective signature from a measurement.
 *
 * @param measurement Measurement vector (typically the OR of one or more signatures).
 * @param signature_matrix signature_matrix[item_id] contains the complete signature.
 * @param defective_item_id Known defective to peel.
 * @return Peeled measurement.
 */
inline vector<bool> peelSignature(
    const vector<bool> &measurement,
    const vector<vector<bool>> &signature_matrix,
    uint defective_item_id)
{
    uint signature_length = measurement.size();
    uint L = ceil(log2(signature_matrix.size()));
    uint num_permutations = signature_length / (2 * L);

    vector<bool> result(signature_length);

    const vector<bool> &defective_signature =
        signature_matrix[defective_item_id];

    for (uint i = 0; i < num_permutations; i++)
    {
        for (uint j = 0; j < L; j++)
        {
            uint bit_id = i * 2 * L + j;
            uint complement_id = bit_id + L;

            // A measurement bit cannot be 0 if the peeled item has a 1.
            assert(!(measurement[bit_id] == 0 &&
                     defective_signature[bit_id] == 1));

            if (measurement[bit_id])
            {
                if (defective_signature[bit_id])
                {
                    result[bit_id] =
                        (measurement[complement_id] == 0);
                }
                else
                {
                    result[bit_id] = 1;
                }
            }
            else
            {
                result[bit_id] = 0;
            }

            result[complement_id] = !result[bit_id];
        }
    }

    return result;
}
/**
 * @brief Generate pool residuals by OR-ing together the signatures of
 * the defective items present in each pool.
 */
inline vector<vector<bool>> generateResiduals(
    const PoolingMatrix &base_pooling_matrix,
    const vector<vector<bool>> &signature_matrix,
    const set<uint> &defectives)
{
    uint signature_length = signature_matrix[0].size();

    vector<vector<bool>> residuals(
        base_pooling_matrix.num_pools,
        vector<bool>(signature_length, false));

    for (uint pool_id = 0;
         pool_id < base_pooling_matrix.num_pools;
         pool_id++)
    {
        for (uint item_id :
             base_pooling_matrix.pools_to_items[pool_id])
        {
            if (defectives.count(item_id) == 0)
                continue;

            for (uint bit = 0;
                 bit < signature_length;
                 bit++)
            {
                residuals[pool_id][bit] =
                    residuals[pool_id][bit] |
                    signature_matrix[item_id][bit];
            }
        }
    }

    return residuals;
}
inline set<uint> peelingAlgorithm(
    vector<vector<bool>> residuals,
    vector<vector<bool>> &signature_matrix,
    set<uint> identified_defectives,
    PoolingMatrix &base_pooling_matrix,
    vector<vector<uint>> &permutation_map,
    int debug = 0)
{
    uint num_pools = base_pooling_matrix.num_pools;

    // unresolved_pools[position[i]] stores unresolved pools.
    vector<uint> unresolved_pools(num_pools);
    vector<uint> position(num_pools);
    vector<bool> status(num_pools, false);

    for (uint i = 0; i < num_pools; i++)
    {
        unresolved_pools[i] = i;
        position[i] = i;
    }

    set<uint> defective_items;

    while (!identified_defectives.empty() &&
           !unresolved_pools.empty())
    {
        uint item_id = *identified_defectives.begin();
        identified_defectives.erase(
            identified_defectives.begin());

        defective_items.insert(item_id);

        for (uint pool_id :
             base_pooling_matrix.items_to_pools[item_id])
        {
            // Skip if already resolved.
            if (status[pool_id])
                continue;

            status[pool_id] = true;

            // Try singleton decoding.
            int code =
                decodeSignature(
                    residuals[pool_id],
                    permutation_map);

            if (code >= 0)
            {
                uint idx = position[pool_id];
                uint moved_pool =
                    unresolved_pools.back();

                position[moved_pool] = idx;

                swap(unresolved_pools[idx],
                     unresolved_pools.back());

                unresolved_pools.pop_back();

                continue;
            }

            // Peel and decode.
            vector<bool> result =
                peelSignature(
                    residuals[pool_id],
                    signature_matrix,
                    item_id);

            code =
                decodeSignature(
                    result,
                    permutation_map);

            if (code >= 0)
            {
                if (defective_items.find(code) ==
                    defective_items.end())
                {
                    defective_items.insert(code);

                    if (identified_defectives.find(code) ==
                        identified_defectives.end())
                    {
                        identified_defectives.insert(code);
                    }
                }

                uint idx = position[pool_id];
                uint moved_pool =
                    unresolved_pools.back();

                position[moved_pool] = idx;

                swap(unresolved_pools[idx],
                     unresolved_pools.back());

                unresolved_pools.pop_back();
            }
            else
            {
                assert(code == -3);

                uint idx = position[pool_id];
                uint moved_pool =
                    unresolved_pools.back();

                position[moved_pool] = idx;

                swap(unresolved_pools[idx],
                     unresolved_pools.back());

                unresolved_pools.pop_back();
            }
        }
    }
    // -----------------------------------------------------------------
    // Phase 2:
    // No identified defectives remain, so repeatedly scan unresolved pools
    // looking for newly exposed singletons.
    // -----------------------------------------------------------------

    while (!unresolved_pools.empty())
    {
        bool progress = false;

        for (uint i = 0; i < unresolved_pools.size();)
        {
            uint pool_id = unresolved_pools[i];

            int code =
                decodeSignature(
                    residuals[pool_id],
                    permutation_map);

            if (code >= 0)
            {
                if (identified_defectives.find(code) ==
                        identified_defectives.end() &&
                    defective_items.find(code) ==
                        defective_items.end())
                {
                    identified_defectives.insert(code);
                }
            }

            // Remove pools that are no longer multitons.
            if (code != -2)
            {
                uint idx = position[pool_id];
                uint moved_pool =
                    unresolved_pools.back();

                position[moved_pool] = idx;

                swap(unresolved_pools[idx],
                     unresolved_pools.back());

                unresolved_pools.pop_back();

                progress = true;

                // Do not increment i because a new pool has
                // been swapped into position i.
                continue;
            }

            ++i;
        }

        // Explore every newly discovered defective.
        while (!identified_defectives.empty())
        {
            uint item_id = *identified_defectives.begin();
            identified_defectives.erase(
                identified_defectives.begin());

            defective_items.insert(item_id);

            for (uint pid :
                 base_pooling_matrix.items_to_pools[item_id])
            {
                if (status[pid])
                    continue;

                status[pid] = true;

                vector<bool> result =
                    peelSignature(
                        residuals[pid],
                        signature_matrix,
                        item_id);

                int code =
                    decodeSignature(
                        result,
                        permutation_map);

                if (code >= 0)
                {
                    if (identified_defectives.find(code) ==
                            identified_defectives.end() &&
                        defective_items.find(code) ==
                            defective_items.end())
                    {
                        identified_defectives.insert(code);
                    }
                }
            }
        }

        // Prevent an infinite loop if nothing changes.
        if (!progress &&
            identified_defectives.empty())
        {
            break;
        }
    }

    // Add any remaining queued defectives.
    while (!identified_defectives.empty())
    {
        uint item_id = *identified_defectives.begin();
        identified_defectives.erase(
            identified_defectives.begin());

        defective_items.insert(item_id);
    }

    return defective_items;
}

set<uint> randomDefectives(uint num_features, uint k)
{
    vector<uint> items(num_features);

    for (uint i = 0; i < num_features; i++)
        items[i] = i;

    static mt19937 rng(42);

    shuffle(items.begin(), items.end(), rng);

    set<uint> defectives;

    for (uint i = 0; i < k; i++)
        defectives.insert(items[i]);

    return defectives;
}
int main()
{

    uint num_features = 100000;
    uint sparsity = 10;

    cout << "========================================\n";
    cout << "Testing computePools()\n";
    cout << "========================================\n\n";

    PoolingMatrix pooling_matrix = computePools(num_features, sparsity);

    cout << "Number of Features : " << pooling_matrix.num_features << '\n';
    cout << "Number of Pools    : " << pooling_matrix.num_pools << "\n\n";

    cout << "Pools -> Items\n";
    cout << "--------------\n";

    for (uint pool_id = 0; pool_id < pooling_matrix.num_pools; pool_id++)
    {

        // cout << "Pool " << pool_id << " : ";

        for (uint item_id : pooling_matrix.pools_to_items[pool_id])
        {
            // cout << item_id << " ";
        }

        cout << '\n';
    }

    cout << '\n';

    cout << "Items -> Pools\n";
    cout << "--------------\n";

    for (uint item_id = 0; item_id < pooling_matrix.num_features; item_id++)
    {

        // cout << "Item " << item_id << " : ";

        for (uint pool_id : pooling_matrix.items_to_pools[item_id])
        {
            // cout << pool_id << " ";
        }

        cout << '\n';
    }

    cout << "\n========================================\n";
    cout << "Testing getPermutationMap()\n";
    cout << "========================================\n\n";

    uint num_permutations = 3;

    vector<vector<uint>> permutation_map =
        getPermutationMap(num_features, num_permutations);

    cout << "Permutation Map (item_id -> signature numbers)\n";
    cout << "----------------------------------------------\n";

    for (uint item_id = 0; item_id < num_features; item_id++)
    {

        // cout << "Item " << item_id << " : ";

        for (uint sign_number : permutation_map[item_id])
        {
            // cout << sign_number << " ";
        }

        cout << '\n';
    }

    cout << "\n========================================\n";
    cout << "Testing getSignature()\n";
    cout << "========================================\n\n";

    for (uint item_id = 0; item_id < num_features; item_id++)
    {

        vector<bool> signature =
            getSignature(permutation_map[item_id], num_features);

        // cout << "Item " << item_id << " : ";
        uint i = 1;
        for (bool bit : signature)
        {
            // cout << bit;
            if (i % 6 == 0)
            {
                cout << "  ";
            }
            i++;
        }

        cout << '\n';
    }

    // cout << "\n========================================\n";
    // cout << "Testing getSignatureMatrix()\n";
    // cout << "========================================\n\n";

    vector<vector<bool>> signature_matrix =
        getSignatureMatrix(permutation_map, num_features);

    // cout << "Signature Matrix\n";
    // cout << "----------------\n";

    // for (uint item_id = 0; item_id < num_features; item_id++)
    // {

    //     cout << "Item " << item_id << " : ";

    //     for (bool bit : signature_matrix[item_id])
    //     {
    //         cout << bit;
    //     }

    //     cout << '\n';
    // }

    cout << "\n========================================\n";
    cout << "Testing getExtendedPoolingMatrix()\n";
    cout << "========================================\n\n";

    PoolingMatrix extended_pooling_matrix =
        getExtendedPoolingMatrix(
            pooling_matrix,
            signature_matrix,
            num_features,
            num_permutations);

    cout << "Number of Features : " << extended_pooling_matrix.num_features << '\n';
    cout << "Number of Pools    : " << extended_pooling_matrix.num_pools << "\n\n";

    cout << "Pools -> Items\n";
    cout << "--------------\n";

    for (uint pool_id = 0;
         pool_id < extended_pooling_matrix.num_pools;
         pool_id++)
    {

        // cout << "Pool " << pool_id << " : ";

        for (uint item_id :
             extended_pooling_matrix.pools_to_items[pool_id])
        {
            // cout << item_id << " ";
        }

        cout << '\n';
    }

    cout << '\n';

    cout << "Items -> Pools\n";
    cout << "--------------\n";

    for (uint item_id = 0;
         item_id < extended_pooling_matrix.num_features;
         item_id++)
    {

        // cout << "Item " << item_id << " : ";

        for (uint pool_id :
             extended_pooling_matrix.items_to_pools[item_id])
        {
            // cout << pool_id << " ";
        }

        cout << '\n';
    }

    cout << "\n========================================\n";
    cout << "Testing decodeBlock()\n";
    cout << "========================================\n\n";

    uint L = ceil(log2(num_features));

    // ---------------- Singleton ----------------

    vector<bool> singleton =
        getBlockSignature(3, num_features);

    cout << "Singleton Block : ";

    for (bool bit : singleton)
        cout << bit;

    cout << "\nDecoded = "
         << decodeBlock(singleton)
         << "\n\n";

    // ---------------- Zeroton ----------------

    vector<bool> zeroton(2 * L, false);

    cout << "Zeroton Block   : ";

    for (bool bit : zeroton)
        cout << bit;

    cout << "\nDecoded = "
         << decodeBlock(zeroton)
         << "\n\n";

    // ---------------- Multiton ----------------

    // Break the complement relationship.
    vector<bool> multiton = singleton;
    multiton[0] = 1;

    cout << "Multiton Block  : ";

    for (bool bit : multiton)
        cout << bit;

    cout << "\nDecoded = "
         << decodeBlock(multiton)
         << "\n";

    cout << "\n========================================\n";
    cout << "Testing decodeSignature()\n";
    cout << "========================================\n\n";

    // ---------------- Decode every signature ----------------

    for (uint item_id = 0; item_id < num_features; item_id++)
    {
        int decoded =
            decodeSignature(signature_matrix[item_id],
                            permutation_map);

        // cout << "Item "
        //      << item_id
        //      << " -> Decoded = "
        //      << decoded;

        if (decoded == static_cast<int>(item_id))
            // cout << "  [PASS]";
        // else
            // cout << "  [FAIL]";

        cout << '\n';
    }

    // ---------------- Zeroton ----------------

    vector<bool> zero_signature(signature_matrix[0].size(), false);

    cout << "\nZeroton -> "
         << decodeSignature(zero_signature,
                            permutation_map)
         << '\n';

    // ---------------- Verification Failure ----------------

    // Corrupt one bit in the second block.
    vector<bool> bad_signature = signature_matrix[0];

    uint block_len = bad_signature.size() / num_permutations;

    // Flip the first bit of block 2.
    bad_signature[block_len] =
        !bad_signature[block_len];

    cout << "Corrupted Signature -> "
         << decodeSignature(bad_signature,
                            permutation_map)
         << '\n';

    cout << "\n========================================\n";
    cout << "Testing peelSignature()\n";
    cout << "========================================\n\n";

    // Choose two items.
    uint itemA = 1;
    uint itemB = 3;

    // Simulate a doubleton measurement using bitwise OR.
    vector<bool> measurement(signature_matrix[0].size());

    for (uint i = 0; i < measurement.size(); i++)
    {
        measurement[i] =
            signature_matrix[itemA][i] |
            signature_matrix[itemB][i];
    }

    cout << "Doubleton = Item "
         << itemA
         << " OR Item "
         << itemB
         << '\n';

    // Peel itemA.
    vector<bool> peeled =
        peelSignature(measurement,
                      signature_matrix,
                      itemA);

    int decoded =
        decodeSignature(peeled,
                        permutation_map);

    cout << "Peeling Item "
         << itemA
         << " leaves Item "
         << decoded;

    if (decoded == static_cast<int>(itemB))
        cout << "  [PASS]";
    else
        cout << "  [FAIL]";

    cout << '\n';

    // Peel itemB.
    peeled =
        peelSignature(measurement,
                      signature_matrix,
                      itemB);

    decoded =
        decodeSignature(peeled,
                        permutation_map);

    cout << "Peeling Item "
         << itemB
         << " leaves Item "
         << decoded;

    if (decoded == static_cast<int>(itemA))
        cout << "  [PASS]";
    else
        cout << "  [FAIL]";

    cout << '\n';

    cout << "\n========================================\n";
    cout << "Testing generateResiduals()\n";
    cout << "========================================\n\n";

    set<uint> defectives = {1, 3};

    vector<vector<bool>> residuals =
        generateResiduals(
            pooling_matrix,
            signature_matrix,
            defectives);

    for (uint pool = 0;
         pool < pooling_matrix.num_pools;
         pool++)
    {
        cout << "Pool "
             << pool
             << " : ";

        for (bool bit : residuals[pool])
            cout << bit;

        cout << '\n';
    }
    cout << "\n========================================\n";
    cout << "Testing peelingAlgorithm()\n";
    cout << "========================================\n\n";

    // Ground truth defective set.
    set<uint> actual_defectives = randomDefectives(num_features,sparsity);

    // Generate synthetic residuals.
     residuals =
        generateResiduals(
            pooling_matrix,
            signature_matrix,
            actual_defectives);

    // Assume one defective has already been identified.
    set<uint> identified_defectives;
    identified_defectives.insert(
    *actual_defectives.begin());

    // Run peeling.
    set<uint> recovered =
        peelingAlgorithm(
            residuals,
            signature_matrix,
            identified_defectives,
            pooling_matrix,
            permutation_map);

    // ----------------------------------------------------
    // Print results
    // ----------------------------------------------------

    cout << "Actual defectives      : ";
    for (uint item : actual_defectives)
        cout << item << " ";
    cout << '\n';

    cout << "Initially identified   : ";
    for (uint item : identified_defectives)
        cout << item << " ";
    cout << '\n';

    cout << "Recovered defectives   : ";
    for (uint item : recovered)
        cout << item << " ";
    cout << '\n';

    // Compare sets.
    if (recovered == actual_defectives)
    {
        cout << "\n[PASS] All defectives recovered.\n";
    }
    else
    {
        cout << "\n[FAIL] Recovery incorrect.\n";

        cout << "Missing: ";
        for (uint item : actual_defectives)
        {
            if (recovered.find(item) == recovered.end())
                cout << item << " ";
        }

        cout << "\nExtra: ";
        for (uint item : recovered)
        {
            if (actual_defectives.find(item) == actual_defectives.end())
                cout << item << " ";
        }

        cout << '\n';
    }
    return 0;
}
