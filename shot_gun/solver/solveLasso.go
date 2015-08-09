package solver

import (
	//	"flag"
	"fmt"
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
func get_obj_2(p *readData.Problem, x []float32) float32 {
	var loss float32
	var l1 float32
	var t float32
	loss = 0
	l1 = 0
	for i := 0; i < p.L; i++ {
		t = 0.0
		for j := 0; j < len(p.A_rows[i].Idxs); j++ {
			index := p.A_rows[i].Idxs[j]
			t = t + p.A_rows[i].Values[j]*x[index] - p.A_rows[i].Values[j]*x[p.N+index]
		}
		t = float32(p.Labels[i]) - t
		loss = loss + t*t
	}
	loss = loss / float32(p.L) / 2.0
	for i := 0; i < p.N*2; i++ {
		l1 = l1 + x[i]
	}
	return loss + l1*p.Lambda
}
func get_obj(p *readData.Problem) float32 {
	var loss float32
	var l1 float32
	var t float32
	loss = 0
	l1 = 0
	for i := 0; i < p.L; i++ {
		t = float32(p.Labels[i]) - p.Ax[i]
		//		t = float32(p.Labels[i]) - p.A_rows[i].Multiply_dense_array(p.X)
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
	//	var beta float32
	//	beta = 1 //beta =1 for lasso
	obj_old := get_obj(p)
	//	obj_old := get_obj_2(p, x)
	fmt.Printf("obj: %f\n", obj_old)

	grad := make([]float32, 2*p.N)
	for i := 0; i < p.N*2; i++ {
		if i < p.N {
			grad[i] = p.Lambda - p.A_cols[i].Multiply_dense_int_array(p.Labels)/float32(p.L)
		} else {
			grad[i] = p.Lambda + p.A_cols[i-p.N].Multiply_dense_int_array(p.Labels)/float32(p.L)
		}
	}
	//	fmt.Println("%v", grad)
	var iter int
	for iter = 0; iter < p.Max_iter; iter++ {
		random_idx := rand.Perm(p.N * 2)
		for j_idx := 0; j_idx < p.N*2; j_idx++ {
			j := random_idx[j_idx]
			d_xj := -x[j]
			if -grad[j] > d_xj {
				d_xj = -grad[j]
			}
			x[j] = x[j] + d_xj
			update_Ax(p, j, d_xj)
			update_grad(p, grad)
			if j_idx%1000 == 0 {
				update_X(p, x)
				obj_new := get_obj(p)
				fmt.Printf("obj: %f %d %f\n", obj_new, j, d_xj)
			}
		}
		update_grad(p, grad)
		update_X(p, x)
		obj_new := get_obj(p)
		fmt.Printf("iter: %d obj: %f\n", iter, obj_new)
		if obj_old-obj_new < obj_new*p.Epsilon {
			fmt.Printf("converged!")
			break
		}
	}
}
func update_Ax(p *readData.Problem, j int, d_xj float32) {
	if j >= p.N {
		for i := 0; i < len(p.A_cols[j-p.N].Idxs); i++ {
			index := p.A_cols[j-p.N].Idxs[i]
			p.Ax[index] = p.Ax[index] - p.A_cols[j-p.N].Values[i]*d_xj
		}
	} else {
		for i := 0; i < len(p.A_cols[j].Idxs); i++ {
			index := p.A_cols[j].Idxs[i]
			p.Ax[index] = p.Ax[index] + p.A_cols[j].Values[i]*d_xj
		}
	}
}
func update_grad(p *readData.Problem, grad []float32) {
	for i := 0; i < p.N; i++ {
		grad[i] = p.A_cols[i].Multiply_dense_array(p.Ax)/float32(p.L) - p.ATy[i]/float32(p.L) + p.Lambda
	}
	for i := p.N; i < p.N*2; i++ {
		grad[i] = -p.A_cols[i-p.N].Multiply_dense_array(p.Ax)/float32(p.L) + p.ATy[i-p.N]/float32(p.L) + p.Lambda
	}
}

//func update_grad(p *readData.Problem, j int, d_xj float32, grad []float32) {
//	if j > p.N {
//		for i := 0; i < p.N; i++ {
//			grad[i] = grad[i] - d_xj*p.A_cols[i].Multiply_sparse_vector(&p.A_cols[j-p.N])
//		}
//		for i := p.N; i < p.N*2; i++ {
//			grad[i] = grad[i] + d_xj*p.A_cols[i-p.N].Multiply_sparse_vector(&p.A_cols[j-p.N])
//		}
//	} else {
//		for i := 0; i < p.N; i++ {
//			grad[i] = grad[i] + d_xj*p.A_cols[i].Multiply_sparse_vector(&p.A_cols[j])
//		}
//		for i := p.N; i < p.N*2; i++ {
//			grad[i] = grad[i] - d_xj*p.A_cols[i-p.N].Multiply_sparse_vector(&p.A_cols[j])
//		}
//	}
//}
func update_X(p *readData.Problem, x []float32) {
	for i := 0; i < p.N; i++ {
		p.X[i] = x[i] - x[p.N+i] //!!
	}
}
