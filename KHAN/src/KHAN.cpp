#include "headers.hpp"
// #include "HashFunction.hpp"
#include "BloomHashFunction.hpp"
// #include "PoolingMatrix.hpp"
// #include "Saffron.hpp"
// #include "ParityBuckets.hpp"
// #include "SaffronIndex.hpp"
// #include "GlobalInvertedIndex.hpp"
// #include "MLGTSaffron.hpp"
// #include "MLGTGlobal.hpp"
// #include "BloomHashIndex.hpp"
// #include "BloomGroupTestingSaffron.hpp"
#include "OrionIndex.hpp"
// #include "NovaMatrix.hpp"
// #include "PulsarSaffron.hpp"
#include "KHAN.hpp"
#include "BruteForceHashSearch.hpp"

namespace py = pybind11;

PYBIND11_MODULE(KHAN, m){
    m.doc() = "mlgt_saffron python module using SAFFRON for fast nearest neighbor search.";

    // Export Constants
    m.attr("NUM_POOLS_COEFF") = NUM_POOLS_COEFF;
    m.attr("POOLS_PER_ITEM") = POOLS_PER_ITEM;
    m.attr("SIGNATURE_COEFF") = SIGNATURE_COEFF;
    // m.attr("NUM_HASH_BITS") = NUM_HASH_BITS;
    m.attr("BLOOM_HASH_BITS") = BLOOM_HASH_BITS;
    m.attr("BLOOM_NUM_HASHES") = BLOOM_NUM_HASHES;
    m.attr("BLOOM_THRESHOLD") = BLOOM_THRESHOLD;

   
    // BloomHashFunction
    py::class_<BloomHashFunction>(m, "BloomHashFunction", "A Bloom Filter-inspired hash function generating multiple compound hash values.")
        .def(py::init<uint, uint, uint, uint, int>(),
             py::arg("dimension"),
             py::arg("num_hashes") = BLOOM_NUM_HASHES,
             py::arg("num_bits") = BLOOM_HASH_BITS,
             py::arg("threshold") = BLOOM_THRESHOLD,
             py::arg("debug") = 0,
             "Initializes a BloomHashFunction.")
        .def("dimension", &BloomHashFunction::dimension, "Returns the input dimensionality.")
        .def("num_hashes", &BloomHashFunction::num_hashes, "Returns the number of compound hash functions.")
        .def("num_bits", &BloomHashFunction::num_bits, "Returns the number of bits per hash.")
        .def("__call__", (vector<uint> (BloomHashFunction::*)(const py::array_t<float>&) const) &BloomHashFunction::operator(), py::arg("point"),
             "Computes multiple hash values for a given point (callable interface, numpy).");

 
    
    
   
     // OrionIndex
     py::class_<OrionIndex>(m, "OrionIndex",
     "Global inverted index organized by pools.")
     .def(py::init<
        uint,
        uint,
        uint,
        uint>(),
        py::arg("num_features"),
        py::arg("num_masks"),
        py::arg("num_hashes"),
        py::arg("threshold"))
     .def(
        "build",
        (void (OrionIndex::*)(
            vector<vector<uint>>&,
            const vector<vector<uint>>&
        )) &OrionIndex::build,
        py::arg("all_hashes"),
        py::arg("item_indices")
    )
    .def(
        "build_masks",
        (void (OrionIndex::*)(
            const vector<vector<bool>>&,
            const MaskMatrix&
        )) &OrionIndex::build,
        py::arg("all_hashes"),
        py::arg("sky_map")
    );

         
    // KHAN
    py::class_<KHAN>(
    m,
    "KHAN",
    "KHAN nearest neighbor search implementation.")
    .def(py::init<
            py::array_t<float>,
            uint,
            uint,
            uint,
            uint,
            double,
            double,
            int,
            bool>(),
         py::arg("data_points"),
         py::arg("num_neighbors") = 100,
         py::arg("num_features"),
         py::arg("num_hashes") = BLOOM_NUM_HASHES,
         py::arg("hash_bits") = BLOOM_HASH_BITS,
         py::arg("num_degrees") = 10,
         py::arg("cover_fraction") = 0.99,
         py::arg("debug") = 0,
         py::arg("normalize") = true)

    .def_readonly("threshold_", &KHAN::threshold_)
    .def_readonly("all_hashes_", &KHAN::all_hashes_)
    .def_readonly("packed_hashes_", &KHAN::packed_hashes_)
    .def_readonly("num_masks_", &KHAN::num_masks_)

    .def("search", &KHAN::search, py::arg("query"))
    .def("bruteSearch", &KHAN::bruteSearch, py::arg("query"))
    .def("__call__", &KHAN::operator(), py::arg("query"));

    
    // Brute Force Search
     py::class_<BruteForceHashSearch>(
     m,
     "BruteForceHashSearch",
     "Brute-force hash search baseline.")
    .def(py::init<
    uint,
    uint,
    uint,
    uint,
    uint,
    int,
    std::vector<std::vector<bool>>,
    std::vector<std::vector<uint64_t>>&
    >(),
    py::arg("threshold"),
    py::arg("num_features"),
    py::arg("dimension"),
    py::arg("num_hashes"),
    py::arg("hash_bits"),
    py::arg("debug") = 0,
    py::arg("all_hashes"),
    py::arg("packed_hashes"))
     .def(
    "bruteSearch",
    py::overload_cast<py::array_t<float>>(
        &BruteForceHashSearch::bruteSearch),
    py::arg("query"))

    .def("bruteSearch",
     py::overload_cast<py::array_t<float>, uint>(
         &BruteForceHashSearch::bruteSearch),
     py::arg("query"),
     py::arg("excess"));
    
    
}

