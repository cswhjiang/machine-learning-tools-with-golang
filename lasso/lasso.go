// lasso
//min 1/(2L) \|y - A x\|^2 + lambda \|x\|_1
package main

import (
	"flag"
	"fmt"
	//	"github.com/cswhjiang/machine-learning-tools-with-golang/utils/mathOperator"
	"github.com/cswhjiang/machine-learning-tools-with-golang/utils/readData"
	"os"
	"time"
)

func main() {
	// /home/jwh/dataset/rcv1/rcv1_train.binary
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

	p, _ := readData.ReadData(train_file_name, false)
	p.Lambda = float32(lambda)
	p.Epsilon = 0.0001
	p.PrintProblem()
	solve_lasso_CD(p)
	elapsed := time.Since(start)

	fmt.Printf("took %s \n", elapsed)

	p_test, _ := readData.ReadData(test_file_name, false)
	p_test.X = p.X
	fmt.Printf("testing...\n")
	loss_test := get_acc_as_reg(p_test)
	fmt.Printf("testing error: %e\n", loss_test)
}

func abs(a float32) float32 {
	if a > 0 {
		return a
	} else {
		return -a
	}
}

func get_acc_as_reg(p *readData.Problem) float32 {
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
		l1 = l1 + abs(p.X[i])
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

func update_residual(residual []float32, z []float32, p *readData.Problem, n int) {
	for i := 0; i < len(p.A_cols[n].Idxs); i++ {
		index := p.A_cols[n].Idxs[i]
		residual[index] = z[index] - p.A_cols[n].Values[i]*p.X[n]
	}
}
func update_z(z []float32, residual []float32, p *readData.Problem, n int) {
	for i := 0; i < len(p.A_cols[n].Idxs); i++ {
		index := p.A_cols[n].Idxs[i]
		z[index] = residual[index] + p.A_cols[n].Values[i]*p.X[n]
	}
}

//compute x_i while fixing the other variable
func solve_lasso_CD(p *readData.Problem) {

	//initial pred is zeros, since x is zero vector, hence the residual is just the label vector
	residual := make([]float32, p.L)
	z := make([]float32, p.L)
	for i := 0; i < p.L; i++ {
		residual[i] = float32(p.Labels[i])
	}
	//	pred := make([]float32, p.L)
	fea_square := make([]float32, p.N)
	for i := 0; i < p.N; i++ {
		fea_square[i] = p.A_cols[i].Multiply_sparse_vector(&(p.A_cols[i]))
	}
	//	fmt.Printf("%v ", fea_square)
	obj_old := get_obj(p)
	var iter int
	for iter = 1; iter < p.Max_iter; iter++ {
		for n := 0; n < p.N; n++ {
			update_z(z, residual, p, n)
			temp := p.A_cols[n].Multiply_dense_array(z) / fea_square[n]
			p.X[n] = soft_threshold(temp, float32(p.L)*p.Lambda)
			update_residual(residual, z, p, n)
		}
		obj_new := get_obj(p)
		fmt.Printf("obj: %f\n", obj_new)
		if abs(obj_new-obj_old) < p.Epsilon {
			break
		}
		obj_old = obj_new
	}
	fmt.Printf("converged in %d iterations\n", iter)
}
