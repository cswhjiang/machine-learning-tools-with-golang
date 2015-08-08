package solver

import (
	//	"flag"
	//	"fmt"
	//	"github.com/cswhjiang/machine-learning-tools-with-golang/logistic_regression/solver"
	"github.com/cswhjiang/machine-learning-tools-with-golang/utils/mathOp"
	"github.com/cswhjiang/machine-learning-tools-with-golang/utils/readData"
	//	"log"
	"math/rand"
)

func Get_loss_as_regression(p *readData.Problem) float32 {
	var loss float32
	for i := 0; i < p.L; i++ {
		pred := p.A_rows[i].Multiply_dense_array(p.X)
		diff := pred - float32(p.Labels[i])
		loss = loss + diff*diff
	}
	loss = loss / float32(p.L)
	return loss
}

//solve lasso with coordinate descent
func get_obj(p *readData.Problem) float32 {
	var loss float32
	var l1 float32
	var t float32
	loss = 0
	l1 = 0
	for i := 0; i < p.L; i++ {
		t = float32(p.Labels[i]) - p.A_rows[i].Multiply_dense_array(p.X)
		loss = loss + t*t
	}
	loss = loss / float32(p.L) / 2.0
	for i := 0; i < p.N; i++ {
		l1 = l1 + mathOp.Abs(p.X[i])
	}
	return loss + l1*p.Lambda
}

// solve |Ax-y| + r|x|_1, ref: shot-gun Algorithm 1
func Solve_lasso_with_scd(p *readData.Problem) {
	x := make([]float32, p.N*2) // parameter
	var beta float32
	beta = 1 //beta =1 for lasso
	obj_old := get_obj(p)

	grad := make([]float32, 2*p.N)
	for i := 0; i < p.N*2; i++ {
		if i < p.N {
			grad[i] = p.Lambda - p.A_rows[i].Multiply_dense_int_array(p.Labels)
		} else {
			grad[i] = p.Lambda + p.A_rows[p.N-i].Multiply_dense_int_array(p.Labels)
		}

	}
	var iter int
	for iter = 0; iter < p.Max_iter; iter++ {
		random_idx := rand.Perm(p.N * 2)
		for j_idx := 0; j_idx < p.N*2; j_idx++ {
			j := random_idx[j_idx]
			d_xj := -x[j]
			if -grad[j]/beta > d_xj {
				d_xj = -grad[j] / beta
			}
			x[j] = x[j] + d_xj
			//update grad
			update_grad(p, j, d_xj, grad)
		}
		update_X(p, x)
		obj_new := get_obj(p)
		if obj_old-obj_new < obj_new*p.Epsilon {
			break
		}
	}
}
func update_grad(p *readData.Problem, j int, d_xj float32, grad []float32) {
	if j > p.N {
		for i := 0; i < p.N; i++ {
			grad[i] = grad[i] - d_xj*p.A_cols[i].Multiply_sparse_vector(&p.A_cols[j])
		}
		for i := p.N; i < p.N*2; i++ {
			grad[i] = grad[i] + d_xj*p.A_cols[p.N-i].Multiply_sparse_vector(&p.A_cols[j])
		}
	} else {
		for i := 0; i < p.N; i++ {
			grad[i] = grad[i] + d_xj*p.A_cols[i].Multiply_sparse_vector(&p.A_cols[j])
		}
		for i := p.N; i < p.N*2; i++ {
			grad[i] = grad[i] - d_xj*p.A_cols[p.N-i].Multiply_sparse_vector(&p.A_cols[j])
		}
	}
}
func update_X(p *readData.Problem, x []float32) {
	for i := 0; i < p.N; i++ {
		p.X[i] = x[p.N+i] - x[i]
	}
}
