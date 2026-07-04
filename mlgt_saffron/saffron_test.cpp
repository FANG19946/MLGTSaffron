#pragma once

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <unordered_set>
#include <vector>

using namespace std;

using uint = unsigned int;

// -----------------------------------------------------------------------------
// Replace these with the same values from your project.
// -----------------------------------------------------------------------------

constexpr uint C_epsilon = 2;
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

int main()
{

    uint num_features = 5;
    uint sparsity = 2;

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

        cout << "Pool " << pool_id << " : ";

        for (uint item_id : pooling_matrix.pools_to_items[pool_id])
        {
            cout << item_id << " ";
        }

        cout << '\n';
    }

    cout << '\n';

    cout << "Items -> Pools\n";
    cout << "--------------\n";

    for (uint item_id = 0; item_id < pooling_matrix.num_features; item_id++)
    {

        cout << "Item " << item_id << " : ";

        for (uint pool_id : pooling_matrix.items_to_pools[item_id])
        {
            cout << pool_id << " ";
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

        cout << "Item " << item_id << " : ";

        for (uint sign_number : permutation_map[item_id])
        {
            cout << sign_number << " ";
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

        cout << "Item " << item_id << " : ";
        uint i = 1;
        for (bool bit : signature)
        {
            cout << bit;
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

        cout << "Pool " << pool_id << " : ";

        for (uint item_id :
             extended_pooling_matrix.pools_to_items[pool_id])
        {
            cout << item_id << " ";
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

        cout << "Item " << item_id << " : ";

        for (uint pool_id :
             extended_pooling_matrix.items_to_pools[item_id])
        {
            cout << pool_id << " ";
        }

        cout << '\n';
    }
    return 0;
}
