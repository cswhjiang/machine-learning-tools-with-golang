// logistic_regression
package main

import (
	"flag"
	"fmt"
	"github.com/cswhjiang/machine-learning-tools-with-golang/utils/mathOperator"
	"github.com/cswhjiang/machine-learning-tools-with-golang/utils/readData"
	"math"
	"os"
	"time"
)

func main() {
	fmt.Printf("logistic_regression only support binary classification for now\n")

	var train_file_name string
	var test_file_name string
	var lambda float64
	flag.StringVar(&train_file_name, "train", "", "training file (libsvm format)")
	flag.StringVar(&test_file_name, "test", "", "testing file (libsvm format)")
	flag.Float64Var(&lambda, "lambda", 0.000001, "trade-off parameter")

	if len(os.Args) <= 3 {
		fmt.Printf("Usage: \n")
		flag.PrintDefaults()
		os.Exit(0)
	}
	flag.Parse()

	start := time.Now()

	p, _ := readData.ReadData(train_file_name, true)
	elapsed_reading := time.Since(start)
	fmt.Printf("took %s to read data \n", elapsed_reading)

	p.PrintProblem()
	p.Lambda = float32(lambda)
	p.Epsilon = 0.0001
	solve_lr_CD(p)
	elapsed := time.Since(start)

	fmt.Printf("took %s \n", elapsed)

	p_test, _ := readData.ReadData(train_file_name, true)
	p_test.X = p.X
	fmt.Printf("testing...")
	loss_test := get_acc(p_test)
	fmt.Printf("acc: %f\n", loss_test)
}

func get_acc(p *readData.Problem) float32 {
	var acc float32
	acc = 0
	for i := 0; i < p.L; i++ {
		pred := p.A_rows[i].Multiply_dense_array(p.X)
		if float32(p.Labels[i])*pred > 0 {
			acc = acc + 1.0
		}
	}
	acc = acc / float32(p.L) * 100.0
	return acc
}

//TODO  speed it up (liblinear)
func solve_lr_CD(p *readData.Problem) {
	obj_old := get_obj_lr(p)
	fmt.Printf("obj: %f\n", obj_old)
	for i := 0; i < p.Max_iter; i++ {
		u, w := get_u_and_w(p)
		solve_weighted_lasso_CD(p, u, w)
		obj_new := get_obj_lr(p)
		fmt.Printf("obj: %f\n", obj_new)
		if mathOperator.Abs(obj_new-obj_old) < 0.000001 {
			break
		}
		obj_old = obj_new
	}
}

func get_obj_lr(p *readData.Problem) float32 {
	var loss float32
	for i := 0; i < p.L; i++ {
		ti := (1.0 + float32(math.Exp(float64(p.A_rows[i].Multiply_dense_array(p.X)*float32(-p.Labels[i])))))
		ti = float32(math.Log(float64(ti)))
		loss = loss + ti
	}
	loss = loss / float32(p.L)
	var l1 float32
	for i := 0; i < p.N; i++ {
		l1 = l1 + mathOperator.Abs(p.X[i])
	}
	return loss + l1*p.Lambda
}
func get_u_and_w(p *readData.Problem) ([]float32, []float32) {
	//	z := make([]float32, p.L)
	w := make([]float32, p.L)
	u := make([]float32, p.L)
	for i := 0; i < p.L; i++ {
		pi := 1.0 / (1.0 + float32(math.Exp(float64(p.A_rows[i].Multiply_dense_array(p.X)*float32(-p.Labels[i])))))
		//		z[i] = float32(p.Labels[i]) * (1.0 - pi) / float32(p.L) // can be zero
		w[i] = pi * (1.0 - pi) / float32(p.L) // can be zero
		//		if w[i] > 1e-9 {
		//			u[i] = p.A_rows[i].Multiply_dense_array(p.X) + z[i]/w[i]
		//		} else {
		//			u[i] = p.A_rows[i].Multiply_dense_array(p.X)
		//		}
		u[i] = p.A_rows[i].Multiply_dense_array(p.X) + float32(p.Labels[i])/pi
	}
	return u, w
}
func get_obj(p *readData.Problem, u []float32, w []float32) float32 {
	var loss float32
	var l1 float32
	var t float32
	loss = 0
	l1 = 0
	for i := 0; i < p.L; i++ {
		t = u[i] - p.A_rows[i].Multiply_dense_array(p.X)
		loss = loss + t*t*w[i]
	}
	for i := 0; i < p.N; i++ {
		l1 = l1 + mathOperator.Abs(p.X[i])
	}
	return loss + l1*p.Lambda
}
func soft_threshold(a float32, lambda float32) float32 {
	var r float32
	r = 0.0
	if a > lambda {
		r = a - lambda
	} else if a < -lambda {
		r = a + lambda
	}
	return r
}

// r = u- A*x and it is updated by r = z-A_nx_n
func weighted_lasso_update_residual(residual []float32, z []float32, p *readData.Problem, n int) {
	//	for i := 0; i < len(residual); i++ {
	//		residual[i] = z[i]
	//	}
	//	copy(residual, z)
	for i := 0; i < len(p.A_cols[n].Idxs); i++ {
		index := p.A_cols[n].Idxs[i]
		residual[index] = z[index] - p.A_cols[n].Values[i]*p.X[n]
	}
}

// z = u - (A*x-A_nx_n) = u - A*x + A_nx_n = r + A_nx_n
func weighted_lasso_update_z(z []float32, residual []float32, p *readData.Problem, n int) {
	//	for i := 0; i < len(residual); i++ {
	//		z[i] = residual[i]
	//	}
	//	copy(z, residual) // slow
	for i := 0; i < len(p.A_cols[n].Idxs); i++ {
		index := p.A_cols[n].Idxs[i]
		z[index] = residual[index] + p.A_cols[n].Values[i]*p.X[n]
	}
}

// using CD to solve weigthed lasso
//compute x_i while fixing the other variable
func solve_weighted_lasso_CD(p *readData.Problem, u []float32, w []float32) {
	//initial pred is zeros, since x is zero vector, hence the residual is just the label vector
	residual := make([]float32, p.L)
	z := make([]float32, p.L)
	for i := 0; i < p.L; i++ {
		residual[i] = u[i] - p.A_rows[i].Multiply_dense_array(p.X)
	}
	copy(z, residual) // can  also be deleted
	//	pred := make([]float32, p.L)
	fea_weithed_norm := make([]float32, p.N) // time consuming
	for i := 0; i < p.N; i++ {
		temp := p.A_cols[i].Dot_product(w)
		fea_weithed_norm[i] = p.A_cols[i].Multiply_sparse_vector(temp)
	}
	//	fmt.Printf("%v ", fea_square)
	obj_old := get_obj(p, u, w)
	fmt.Printf("    inner obj: %f\n", obj_old)
	var iter int
	for iter = 1; iter < 100; iter++ {
		for n := 0; n < p.N; n++ {
			weighted_lasso_update_z(z, residual, p, n)
			//only nonzeros entries of z are used
			temp := p.A_cols[n].Multiply_dense_array_weithted(z, w) / fea_weithed_norm[n]
			p.X[n] = soft_threshold(temp, p.Lambda/fea_weithed_norm[n]/2.0)
			weighted_lasso_update_residual(residual, z, p, n)
		}
		obj_new := get_obj(p, u, w)
		if obj_new > obj_old {
			fmt.Printf("    wrong\n")
		}
		fmt.Printf("    inner obj: %f\n", obj_new)
		if mathOperator.Abs(obj_new-obj_old) < 0.0001 {
			break
		}
		obj_old = obj_new
	}
	fmt.Printf("   inner converged in %d iterations\n", iter)
}
