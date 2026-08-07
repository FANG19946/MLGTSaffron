#ifndef OE6D4A82_31B5_4F0C_A8E1_7C9D2B5F64A3
#define OE6D4A82_31B5_4F0C_A8E1_7C9D2B5F64A3


#include <atomic>
#include <tuple>
#include "headers.hpp"
#include "BoolBloomHashFunction.hpp"
#include "MaskMatrix.hpp"
#include "OrionIndex.hpp"
#include <boost/math/distributions/normal.hpp>
#include <bit>
#include <numbers>

/**
 * @brief K-Hypercube Hash Approximate Neighbor implementation.
 * 
 * Uses binary LSH hashes combined with Random Masks and an Inverted Index.
 */
class KHAN  {
public:
    BoolBloomHashFunction shared_hasher_; 
    OrionIndex heliosIndex_; 
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> data_eigen_;
    uint num_hashes_;
    uint hash_bits_;
    uint threshold_;
    uint dimension_;
    bool normalize_; 
    double num_degrees_;
    double cover_fraction_;
    uint num_masks_;
    uint mask_size_;
    uint num_features_;
    uint debug_;
    MaskMatrix sky_map_;
    vector<vector<bool>> all_hashes_;
    vector<vector<uint64_t>> packed_hashes_;
    uint num_packed_hashes_;


public:
    /**
     * @brief Initializes the KHAN index.
     */
    

    KHAN(
        pybind11::array_t<float> data_points_arr,
        uint num_neighbors,
        uint num_features,
        uint num_hashes = BLOOM_NUM_HASHES,
        uint hash_bits = BLOOM_HASH_BITS,
        double num_degrees = 10,
        double cover_fraction = 0.99,
        int debug = 0,
        bool normalize = true
    ) :
        num_hashes_(num_hashes),
        hash_bits_(hash_bits),
        num_degrees_(num_degrees),
        cover_fraction_(cover_fraction),
        dimension_(data_points_arr.shape(1)),
        normalize_(normalize),
        num_features_(num_features),
        debug_(debug),
        all_hashes_(num_features_)
    {



        // Set Mask Size
        assert(num_features_ > 0);
        uint L = ceil(log2(num_features));
        if(C_mask_bits < 0){
            int x = -L;
            assert(C_mask_bits > x);
        }
        L = L + C_mask_bits;
        mask_size_ = L;

        assert(0.0 <= cover_fraction_ && cover_fraction_ <= 1.0);
        assert(0.0 <= num_degrees_ && num_degrees_ <= 180.0);

        // Calculate Expected hamming distance and standard deviation
        cout << "Calulating Expected Hamming Distance and Standard Deviation" << endl;
        double prob_different_bit = num_degrees_/ 180.0 ; 
        double expectation = num_hashes_ * prob_different_bit;
        double std_deviation = std::sqrt(expectation * (1.0 - prob_different_bit));

        // Reduce to standard normal bound
        boost::math::normal normal_dist;
        double z = boost::math::quantile(normal_dist, cover_fraction_);
        double bound = z * std_deviation + expectation;
        threshold_ = ceil(bound);
        if(debug_){
            cout << "Hamming Ball Size: " << threshold_ << endl;
        }
        shared_hasher_ = BoolBloomHashFunction(data_points_arr.shape(1), num_hashes_, hash_bits_, threshold_, debug_);


        // Calculate Number of Masks required 
        cout << "Calculating Number of Masks Required" << endl;
        // Coverage by single mask
        long double mask_coverage = binomial(num_hashes_ - mask_size_, threshold_);
        long double total_sets = binomial(num_hashes_, threshold_);
        long double prob_single_set_covered = mask_coverage / total_sets;
        long double prob_single_set_not_covered = 1.0L - prob_single_set_covered;

        // Number of masks required
        long double required_masks = log((1.0L - cover_fraction_) / binomial(num_hashes_, threshold_))/ log(prob_single_set_not_covered);
        num_masks_ =  static_cast<uint>(std::ceil(required_masks));
        if(debug_){
            cout<< "Coverage by Single Mask: " << prob_single_set_covered << endl;
            cout<< "Number of Masks Required: " << num_masks_ << endl;
        }

        // Building the Mask Matrix
        cout << "Building the Mask Matrix" << endl;
        sky_map_ = MaskMatrix(num_features_, num_hashes_, num_masks_, mask_size_);

        
        
        




        // Convert to Eigen Matrix
        cout<<"Convert to Eigen Matrix"<<endl;
        auto r = data_points_arr.unchecked<2>();
        data_eigen_.resize(r.shape(0), r.shape(1));
        #pragma omp parallel for
        for (int i = 0; i < (int)r.shape(0); ++i) {
            float norm_sq = 0;
            for (int j = 0; j < (int)r.shape(1); ++j) {
                float val = r(i, j);
                data_eigen_(i, j) = val;
                norm_sq += val * val;
            }
            if (normalize_) {
                float norm = std::sqrt(norm_sq);
                if (norm > 1e-9) data_eigen_.row(i) /= norm;
            }
        }


        std::atomic<uint64_t> processed = 0;
        // Pre-calculate hashes for all items
        cout<<"Pre-calculate hashes for all items"<<endl;
        #pragma omp parallel for
        for (int item_idx = 0; item_idx < (int)num_features_; ++item_idx) {
            all_hashes_[item_idx] = shared_hasher_(data_eigen_.row(item_idx));

            // Loader
            uint64_t cnt = ++processed;
            if (cnt % 10000 == 0 || cnt == num_features_) {
                #pragma omp critical
                {
                    std::cout << "\rProcessed " << cnt << " / " << num_features_
                            << " (" << std::fixed << std::setprecision(1)
                            << (100.0 * cnt / num_features_) << "%)"
                            << std::flush;
                }
            }
        }
        std::cout << std::endl;

        num_packed_hashes_ = (num_hashes_ + 63) / 64;
        packed_hashes_.resize(num_features);

        // Pack computed boolean hashes into 64 bit uints
        cout << "Packing Boolean Values into 64 bit Chunks.." << endl;
        #pragma omp parallel for 
        for (int item_idx = 0; item_idx < (int)num_features_; ++item_idx) {
            packed_hashes_[item_idx] = pack_hash(all_hashes_[item_idx]);
        }
        

        // cout << "Loading precomputed hashes..." << endl;
        // vector<vector<uint>> all_hashes(num_features_);

        
        // std::ifstream file("/home/adnan/projects/MLGTSaffron/results/all_hashes_2.csv");
        // if (!file.is_open()) {
        //     std::cerr << "Failed to open file!" << std::endl;
        //     std::exit(1);
        // }

        // std::string line;
        // for (uint i = 0; i < num_features_ && std::getline(file, line); i++) {
        //     std::stringstream ss(line);
        //     std::string value;

        //     while (std::getline(ss, value, ',')) {
        //         all_hashes[i].push_back(std::stoul(value));
        //     }
        // }
        // cout << "Finished loading hashes." << endl;
        


        // Dump hashes to CSV
        // {
        //     std::ofstream out("results/all_hashes_2.csv");

        //     for (const auto& hashes : all_hashes) {
        //         for (size_t i = 0; i < hashes.size(); i++) {
        //             if (i) out << ",";
        //             out << hashes[i];
        //         }
        //         out << "\n";
        //     }
        // }

        // cout << "Saved all_hashes to results/all_hashes_2.csv" << endl;


        
        

        // Build heliosIndex_
        cout<<"Building OrionIndex"<<endl;
        
        
        heliosIndex_ = OrionIndex(num_features_, num_masks_, num_hashes_, threshold_);
        // Index uses the mask matrix
        heliosIndex_.build(all_hashes_, sky_map_);
        cout<<"OrionIndex Built"<<endl;
                

        if (debug_ > 0) {
            cout << "[KHAN] Number of masks = "
                << num_masks_
                << endl;
        }
        
        if (debug_ > 0) {
            cout << "[KHAN] Built " << num_masks_ << " masks." << endl;
        }
    }

    ~KHAN() = default;

protected:
    /**
     * @brief Computes test residuals by identifying candidate items in each pool.
     * 
     * Uses the per-pool inversion indices to find items likely to be similar to the 
     * query, then combines their pre-computed signatures to form the pool residual.
     * 
     * @param query_vec Normalized query vector (Eigen).
     * @return vector<vector<bool>> The num_pools x signature_bits residual matrix, set<uint> identified_defectives contains the set of items that are defectives, double hashing_time Time taken to hash th query vector, double test_evaluation_time time taken to evaluate all the tests parallely, uint total_postings_traversed the total number of postings traversed for all pools.
     */


public:
    /**
     * @brief Performs a nearest neighbor search.
     * 
     * @param query_arr The query vector (numpy array).
     * @return tuple[ vector<uint>, double, double, double ] The item_ids of the defective items, hashing_time, probing_time, verification_time.
     */
    // Changed search to return (topK, hashing_time, decoding_time, total_postings_traversed)
    inline std::tuple<std::vector<uint>, double, double, double, double> search(pybind11::array_t<float> query_arr) {
        Eigen::Map<const Eigen::VectorXf> q_raw(query_arr.data(), dimension_);
        Eigen::VectorXf query = q_raw;
        if (normalize_) {
            float norm = query.norm();
            if (norm > 1e-9) query /= norm;
        }

        // Hashing Time
        auto t_hash_start = std::chrono::high_resolution_clock::now();
        vector<bool> query_hash = shared_hasher_(query);
        auto t_hash_end = std::chrono::high_resolution_clock::now();
        double hashing_time = std::chrono::duration<double>(t_hash_end - t_hash_start).count();
            
        // Probing Time
        auto t_probe_start = std::chrono::high_resolution_clock::now();
        vector<uint> candidate_neighbors = heliosIndex_.get_matches(query_hash, sky_map_);
        auto t_probe_end = std::chrono::high_resolution_clock::now();
        double probing_time = std::chrono::duration<double>(t_probe_end - t_probe_start).count();

        // Verification Time
        auto t_verification_start = std::chrono::high_resolution_clock::now();
        auto [verified_neighbors, candidate_rejection_rate] = verify_neighbors(query_hash, candidate_neighbors, num_hashes_ % 64);
        auto t_verification_end = std::chrono::high_resolution_clock::now();
        double verification_time = std::chrono::duration<double>(t_verification_end - t_verification_start).count();

        
        

        return { verified_neighbors, hashing_time, probing_time, verification_time, candidate_rejection_rate };
    }

    /**
     * @brief Verifies identified items and discards false positives.
     * 
     * @param query_hash vector<bool> The binary hashes of the query vector.
     * @param candidate_neighbors set<uint> The item_ids of the candidate_neighbors.
     * @return tuple[ vector<uint>, double, double, double ] The item_ids of the defective items, hashing_time, probing_time, verification_time.
     */
    inline set<uint> verify_neighbors(const vector<bool> &query_hash,const set<uint> &candidate_neighbors){
        set<uint> verified_neighbors;

        for(const uint &item_id : candidate_neighbors ){

            const auto& item_hash = all_hashes_[item_id];
            assert(query_hash.size() == item_hash.size());
            uint hamming_distance = 0;

            for(uint pos = 0; pos < query_hash.size(); pos++){
                if(query_hash[pos] != item_hash[pos])
                    hamming_distance++;
            }

            if(hamming_distance <= threshold_){
                verified_neighbors.insert(item_id);
            }
        }

        return verified_neighbors;
    }

    inline std::tuple<vector<uint>, double> verify_neighbors(const vector<bool> &query_hash,const vector<uint> &candidate_neighbors, uint excess){
        vector<uint> verified_neighbors;
        verified_neighbors.reserve(candidate_neighbors.size());
        vector<uint64_t> packed_query_hash = pack_hash(query_hash);
        
        double false_candidates = 0.0;
        for(const uint &item_id : candidate_neighbors ){
            uint hamming_distance = 0;
            
            for(uint i = 0; i < num_packed_hashes_; i++){
                hamming_distance += std::popcount(packed_query_hash[i] ^ packed_hashes_[item_id][i]);
                if(hamming_distance > threshold_ + excess)
                    break;
            }
            if(hamming_distance <= threshold_ + excess){
                verified_neighbors.push_back(item_id);
            }
            else
                false_candidates += 1.0;

        }
        double candidate_rejection_rate = 0.0;
        if (!candidate_neighbors.empty())
            candidate_rejection_rate = false_candidates / candidate_neighbors.size();
        

        return {verified_neighbors, candidate_rejection_rate};
    }

    inline std::tuple<vector<uint>, double> verify_neighbors(const Eigen::VectorXf &query, const vector<uint> &candidate_neighbors){
        vector<uint> verified_neighbors;
        verified_neighbors.reserve(candidate_neighbors.size());
       
        
        double false_candidates = 0.0;
        double candidate_rejection_rate = 0.0;

        double cosine_threshold = std::cos(num_degrees_ * std::numbers::pi / 180.0);
        for(const uint &item_id : candidate_neighbors ){
            double similarity = query.dot(data_eigen_.row(item_id));
            candidate_rejection_rate = false_candidates / candidate_neighbors.size();
            if (similarity >= cosine_threshold) {
                verified_neighbors.push_back(item_id);
            }
            else
                false_candidates += 1.0;
        }
        
          
            
            if (!candidate_neighbors.empty())
                candidate_rejection_rate = false_candidates / candidate_neighbors.size();
        

        return {verified_neighbors, candidate_rejection_rate};
    }


    /**
     * @brief Packs boolean hashes into 64 bit uints
     * 
     * @param binary_hash vector<bool> The binary hashes of an item.
     * @return vector<uint64_t> Binary hashes packed into 64 bit uints.
     */
    vector<uint64_t> pack_hash(const vector<bool>& binary_hash) {
        uint num_hashes = binary_hash.size();
        uint num_packed_hashes = (num_hashes + 63) / 64;

        vector<uint64_t> packed_hash(num_packed_hashes);

        for (uint chunk = 0; chunk < num_hashes; chunk += 64) {
            uint64_t base = 1ULL << 63;
            uint64_t hash_val = 0;

            uint end = std::min(chunk + 64, num_hashes);

            for (uint pos = chunk; pos < end; ++pos) {
                if (binary_hash[pos])
                    hash_val |= base;
                base >>= 1;
            }

            packed_hash[chunk / 64] = hash_val;
        }

        return packed_hash;
    }

    /**
     * @brief Performs a search using a callable interface.
     * @param query_arr The query vector.
     * @return vector<uint> Top K item indices.
     */
    inline vector<uint> operator()(pybind11::array_t<float> query_arr) {
        return std::get<0>(search(query_arr));
    }

    /**
     * @brief Performs a hamming ball volume calculation.
     * @param n The number of total bits.
     * @param r The size of the hamming ball.
     * @return long double size of hamming ball.
     */
     inline long double hamming_ball_vol(uint n, uint r){
        long double term = 1;
        long double sum = term;

        for (uint k = 0; k < r; ++k) {
            term *= static_cast<long double>(n - k) / (k + 1);
            sum += term;
        }
        return sum;
     }


    /**
     * @brief Performs nCk.
     * @param n n value in nCk.
     * @param k k value in nCk.
     * @return nCk.
     */ 
    long double binomial(uint n, uint k) {
        if (k > n) return 0.0L;
        if (k > n - k) k = n - k;

        long double result = 1.0L;

        for (uint i = 1; i <= k; ++i) {
            result *= static_cast<long double>(n - k + i);
            result /= i;
        }

        return result;
    }

};



#endif // OE6D4A82_31B5_4F0C_A8E1_7C9D2B5F64A3
