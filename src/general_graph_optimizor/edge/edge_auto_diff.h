#ifndef _GENERAL_GRAPH_OPTIMIZOR_AUTO_DIFF_EDGE_H_
#define _GENERAL_GRAPH_OPTIMIZOR_AUTO_DIFF_EDGE_H_

#include "edge.h"
#include "jet.h"
#include "utility"

namespace slam_solver {

/**
 * @brief Recursive template to calculate the sum of a list of integer dimensions.
 * @tparam Dims Parameter pack of dimension values.
 */
template <int... Dims>
struct TotalDim;

/**
 * @brief Recursive case: sum the first dimension and the sum of the rest.
 * @tparam First The first dimension value.
 * @tparam Rest Remaining dimension values.
 */
template <int First, int... Rest>
struct TotalDim<First, Rest...> {
    static constexpr int value = First + TotalDim<Rest...>::value;
};

/**
 * @brief Base case: sum of empty dimension list is 0.
 */
template <>
struct TotalDim<> {
    static constexpr int value = 0;
};

/**
 * @brief Base class for user-defined cost functions to be used with EdgeAutoDiff.
 * Users must inherit from this class and implement the Evaluate() method.
 * @tparam Scalar Numeric type (float/double) for computation.
 * @tparam ResidualDim Dimension of the output residual vector.
 * @tparam VertexDims Parameter pack of dimensions for each connected vertex.
 */
template <typename Scalar, int ResidualDim, int... VertexDims>
struct CostFunction {
    static constexpr int kResidualDim = ResidualDim;
    static constexpr int kNumVertices = sizeof...(VertexDims);
    static constexpr int kVertexDims[] = {VertexDims...};

    virtual ~CostFunction() = default;

    /**
     * @brief Function call operator, forwards to the Evaluate method.
     * Works for both Scalar (residual calculation) and Jet (Jacobian calculation).
     * @tparam T Type of the parameters (Scalar or Jet).
     * @param parameters Array of pointers to each vertex's parameter data.
     * @param residuals Output array to store computed residuals.
     * @return True if evaluation is successful.
     */
    template <typename T>
    bool operator()(const T* const* parameters, T* residuals) const {
        return Evaluate(parameters, residuals);
    }

    /**
     * @brief Pure virtual method for user to implement the cost function logic.
     * @tparam T Type of the parameters (Scalar or Jet).
     * @param parameters Array of pointers to each vertex's parameter data.
     * @param residuals Output array to store computed residuals.
     * @return True if evaluation is successful.
     */
    template <typename T>
    bool Evaluate(const T* const* parameters, T* residuals) const {
        return false;
    }
};

/**
 * @brief Edge type that computes Jacobians automatically using forward-mode automatic differentiation.
 * Eliminates the need for manual Jacobian derivation.
 * @tparam Functor User-defined cost functor derived from CostFunction.
 * @tparam Scalar Numeric type (float/double).
 * @tparam ResidualDim Dimension of the residual vector.
 * @tparam VertexDims Parameter pack of parameter dimensions for each connected vertex.
 */
template <typename Functor, typename Scalar, int ResidualDim, int... VertexDims>
class EdgeAutoDiff : public Edge<Scalar> {
public:
    /**
     * @brief Constructor, initializes the base Edge and the cost functor.
     * @param functor Instance of the user-defined cost function.
     */
    EdgeAutoDiff(Functor* functor)
        : Edge<Scalar>(ResidualDim, sizeof...(VertexDims)), functor_(functor) {}

    virtual ~EdgeAutoDiff() = default;

    /**
     * @brief Compute the residual vector using standard scalar values.
     * Overrides the base Edge's pure virtual method.
     */
    virtual void ComputeResidual() override {
        const int num_vertices = sizeof...(VertexDims);
        const Scalar* parameters[num_vertices];
        // Collect parameter pointers from all connected vertices
        for (int i = 0; i < num_vertices; ++i) {
            parameters[i] = this->GetVertex(i)->param().data();
        }
        // Execute the cost function to compute residuals
        (*functor_)(parameters, this->residual().data());
    }

    /**
     * @brief Compute the Jacobian matrices for all connected vertices using auto-differentiation.
     * Overrides the base Edge's pure virtual method.
     */
    virtual void ComputeJacobians() override {
        // Compute Jacobian for each vertex using compile-time index sequence
        ComputeAllJacobians(std::make_index_sequence<sizeof...(VertexDims)>{});
    }

private:
    /**
     * @brief Unpack the index sequence and compute Jacobian for each vertex.
     * @tparam Is Compile-time indices for each vertex.
     */
    template <size_t... Is>
    void ComputeAllJacobians(std::index_sequence<Is...>) {
        (ComputeVertexJacobian<Is>(), ...);
    }

    /**
     * @brief Core function: compute Jacobian for a specific vertex using Jet automatic differentiation.
     * @tparam VertexIdx Index of the target vertex to compute Jacobian for.
     */
    template <size_t VertexIdx>
    void ComputeVertexJacobian() {
        // Get the parameter dimension of the target vertex at compile time
        constexpr int current_vertex_param_dim = GetDim<VertexIdx, VertexDims...>::value;
        // Define Jet type with derivative dimension matching the target vertex
        using JetT = Jet<Scalar, current_vertex_param_dim>;

        const JetT* jet_parameters[sizeof...(VertexDims)];
        // Storage to manage lifetime of Jet parameter arrays
        std::vector<std::unique_ptr<JetT[]>> jet_params_storage;

        // 1. Prepare Jet-type parameters for all vertices
        for (size_t v = 0; v < sizeof...(VertexDims); ++v) {
            const auto& v_param = this->GetVertex(v)->param();
            const int v_dim = v_param.rows();

            auto v_jet_ptr = std::make_unique<JetT[]>(v_dim);
            if (v == VertexIdx) {
                // For the target vertex: initialize Jet with derivative seeds (v[j] = 1 for j-th parameter)
                for (int j = 0; j < v_dim; ++j) {
                    v_jet_ptr[j] = JetT(v_param(j), j);
                }
            } else {
                // For other vertices: initialize Jet with zero derivatives (constants)
                for (int j = 0; j < v_dim; ++j) {
                    v_jet_ptr[j] = JetT(v_param(j));
                }
            }
            jet_parameters[v] = v_jet_ptr.get();
            jet_params_storage.push_back(std::move(v_jet_ptr));
        }

        // 2. Evaluate the cost function with Jet variables to compute derivatives
        JetT jet_residuals[ResidualDim];
        (*functor_)(jet_parameters, jet_residuals);

        // 3. Extract the Jacobian matrix (dr/dx) from the infinitesimal part of Jet residuals
        TMat<Scalar> jacobian_param(ResidualDim, current_vertex_param_dim);
        for (int r = 0; r < ResidualDim; ++r) {
            for (int c = 0; c < current_vertex_param_dim; ++c) {
                jacobian_param(r, c) = jet_residuals[r].v(c);
            }
        }

        // 4. Apply manifold chain rule: dr/d_delta = dr/dx * dx/d_delta
        TMat<Scalar> manifold_jacobian;
        this->GetVertex(VertexIdx)->ComputeManifoldJacobian(manifold_jacobian);
        // Store the final Jacobian for the solver
        this->GetJacobian(VertexIdx) = jacobian_param * manifold_jacobian;
    }

    /**
     * @brief Recursive template to get the dimension of the vertex at a given index.
     * @tparam Idx Compile-time vertex index.
     * @tparam Dims Parameter pack of vertex dimensions.
     */
    template <size_t Idx, int... Dims>
    struct GetDim;

    /**
     * @brief Base case: get dimension of the 0-th vertex.
     */
    template <int First, int... Rest>
    struct GetDim<0, First, Rest...> {
        static constexpr int value = First;
    };

    /**
     * @brief Recursive case: decrement index and get dimension from remaining parameter pack.
     */
    template <size_t Idx, int First, int... Rest>
    struct GetDim<Idx, First, Rest...> {
        static constexpr int value = GetDim<Idx - 1, Rest...>::value;
    };

private:
    std::unique_ptr<Functor> functor_; // User-defined cost function functor
};

} // namespace slam_solver

#endif // _GENERAL_GRAPH_OPTIMIZOR_AUTO_DIFF_EDGE_H_
