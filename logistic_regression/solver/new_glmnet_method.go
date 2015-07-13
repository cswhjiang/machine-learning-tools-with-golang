package solver

import (
	//	"flag"
	"fmt"
	"github.com/cswhjiang/machine-learning-tools-with-golang/utils/mathOp"
	"github.com/cswhjiang/machine-learning-tools-with-golang/utils/readData"
	//	"log"
	"math"
	//	"os"
	//	"runtime/pprof"
	//	"time"
)

//0<sigma<1, 0<r<1
func Solve_lr_new_glmnet_cdn(p *readData.Problem, sigma float32, lambda float32) {
	obj_old := get_obj_lr(p)
	fmt.Printf("iter: 0, obj: %f\n", obj_old)

	for i := 1; i <= p.Max_iter; i++ {
		// shuffle should be added here
		for j := 0; j < p.N; j++ {
			if j%10000 == 0 {
				//				fmt.Printf("j: %d    g_j = %f, H_jj = %f, d=%f\n", j, g_j, H_jj, lambda*d)
				obj_new := get_obj_lr(p)
				fmt.Printf("    iter: %d, obj: %f\n", j, obj_new)
			}
			if len(p.A_cols[i].Idxs) == 0 {
				continue
			}
			g_j, H_jj := get_g_j_and_H_jj(p, j)
			if H_jj < 1e-8 || mathOp.Abs(g_j) < 1e-8 {
				continue
			}

			d := update_d(g_j, H_jj, p.X[j], p.Lambda)
			if mathOp.Abs(d) < 1e-8 {
				continue
			}
			//			fmt.Printf("j: %d    g_j = %f, H_jj = %f, d=%f\n", j, g_j, H_jj, d)
			//			loss_old := get_obj_lr(p)
			var r float32
			r = 1.0
			right_hand_size := g_j*d + p.Lambda*(mathOp.Abs(p.X[j]+d)-mathOp.Abs(p.X[j]))

			//compute upper bound of left hand side of (45)
			var upper_1 float32
			var upper_2 float32
			var upper float32
			var s1 float32
			var s2 float32
			var ss1 float32
			var ss2 float32
			var s_y_1 float32
			var s_y_0 float32 //for label -1
			var ss_y_1 float32
			var ss_y_0 float32
			nnz := len(p.A_cols[j].Idxs)

			for i := 0; i < len(p.A_cols[j].Idxs); i++ {
				sample_index := p.A_cols[j].Idxs[i]
				s1 = s1 + p.A_cols[j].Values[i]/(1.0+(mathOp.Exp(p.Ax[sample_index])))
				s2 = s2 + p.A_cols[j].Values[i]/(1.0+(mathOp.Exp(-p.Ax[sample_index])))
				if p.Labels[sample_index] == 1 {
					s_y_1 = s_y_1 + p.A_cols[j].Values[i]
				} else {
					s_y_0 = s_y_0 + p.A_cols[j].Values[i]
				}
			}
			s1 = s1 / float32(p.L)
			s2 = s2 / float32(p.L)
			for {
				//				loss_new := get_obj_with_d(p, d, j, r) //expensive
				ss1 = s1 * (mathOp.Exp(-d*r*p.Xj_max[j]) - 1.0)
				ss1 = ss1 / p.Xj_max[j] * float32(nnz) / float32(p.L)
				upper_1 = float32(nnz) / float32(p.L) * mathOp.Log(1.0+ss1)
				ss_y_0 = d*r*s_y_0/float32(p.L) + p.Lambda*(mathOp.Abs(p.X[j]+d*r)-mathOp.Abs(p.X[j]))
				upper_1 = upper_1 + ss_y_0

				ss2 = s2 * (mathOp.Exp(d*r*p.Xj_max[j]) - 1.0)
				ss2 = ss2 / p.Xj_max[j] * float32(nnz) / float32(p.L)
				upper_2 = float32(nnz) / float32(p.L) * mathOp.Log(1.0+ss2)
				ss_y_1 = -d*r*s_y_1/float32(p.L) + p.Lambda*(mathOp.Abs(p.X[j]+d*r)-mathOp.Abs(p.X[j]))
				upper_2 = upper_2 + ss_y_1

				if upper_1 < upper_2 {
					upper = upper_1
				} else {
					upper = upper_2
				}
				//				upper := get_upper_bound(p, d*r, j)
				if upper < sigma*r*right_hand_size {
					break
				}
				r = r * lambda
				//				fmt.Printf("r=%f\n", r)
			}

			update_Ax(p, p.X[j], p.X[j]+r*d, j)
			p.X[j] = p.X[j] + r*d
			//			if j%1000 == 0 {
			//				//				fmt.Printf("j: %d    g_j = %f, H_jj = %f, d=%f\n", j, g_j, H_jj, lambda*d)
			//				obj_new := get_obj_lr(p)
			//				fmt.Printf("    iter: %d, obj: %f\n", j, obj_new)
			//			}

		}

		obj_new := get_obj_lr(p)
		//		if i%10 == 0 {
		fmt.Printf("iter: %d, obj: %f\n", i, obj_new)
		//		}
		if mathOp.Abs(obj_new-obj_old) < p.Epsilon*obj_old {
			break
		}
		obj_old = obj_new
	}
}

//ref LIBLINEAR -- A Library for Large Linear Classification
func get_upper_bound(p *readData.Problem, dr float32, j int) float32 {
	var upper_1 float32
	var upper_2 float32
	var s1 float32
	var s2 float32
	var s_y_1 float32
	var s_y_0 float32 //for label -1
	nnz := len(p.A_cols[j].Idxs)
	for i := 0; i < len(p.A_cols[j].Idxs); i++ {
		sample_index := p.A_cols[j].Idxs[i]
		s1 = s1 + p.A_cols[j].Values[i]/(1.0+(mathOp.Exp(p.Ax[sample_index])))
		s2 = s2 + p.A_cols[j].Values[i]/(1.0+(mathOp.Exp(-p.Ax[sample_index])))
		if p.Labels[sample_index] == 1 {
			s_y_1 = s_y_1 + p.A_cols[j].Values[i]
		} else {
			s_y_0 = s_y_0 + p.A_cols[j].Values[i]
		}
	}
	s1 = s1 / float32(p.L)
	s1 = s1 * (mathOp.Exp(-dr*p.Xj_max[j]) - 1.0)
	s1 = s1 / p.Xj_max[j] * float32(nnz) / float32(p.L)
	upper_1 = float32(nnz) / float32(p.L) * (mathOp.Log(1.0 + s1))
	s_y_0 = dr*s_y_0/float32(p.L) + p.Lambda*(mathOp.Abs(p.X[j]+dr)-mathOp.Abs(p.X[j]))
	upper_1 = upper_1 + s_y_0

	s2 = s2 / float32(p.L)
	s2 = s2 * (mathOp.Exp(dr*p.Xj_max[j]) - 1.0)
	s2 = s2 / p.Xj_max[j] * float32(nnz) / float32(p.L)
	upper_2 = float32(nnz) / float32(p.L) * (mathOp.Log((1.0 + s2)))
	s_y_1 = -dr*s_y_1/float32(p.L) + p.Lambda*(mathOp.Abs(p.X[j]+dr)-mathOp.Abs(p.X[j]))
	upper_2 = upper_2 + s_y_1

	if upper_1 < upper_2 {
		return upper_1
	} else {
		return upper_2
	}
}

//very expensive
func get_obj_with_d(p *readData.Problem, d float32, j int, r float32) float32 {
	//update p.Ax
	xj_backup := p.X[j]
	xj_new := xj_backup + r*d
	update_Ax(p, xj_backup, xj_new, j)
	obj := get_obj_lr(p)

	//recover p.Ax
	update_Ax(p, xj_new, xj_backup, j)
	return obj
}

func update_d(g_j float32, H_jj float32, xj float32, lambda float32) (d float32) {
	if H_jj > g_j+lambda {
		d = -(g_j + lambda) / H_jj
	} else if H_jj < g_j-lambda {
		d = -(g_j - lambda) / H_jj
	} else {
		d = -xj
	}
	return d
}
func get_g_j_and_H_jj(p *readData.Problem, j int) (float32, float32) {
	var g_j float32
	var H_jj float32
	g_j = 0
	H_jj = 0
	for i := 0; i < len(p.A_cols[j].Idxs); i++ {
		sample_index := p.A_cols[j].Idxs[i]
		//		fmt.Printf("sample: %d, j:%d ", sample_index, j)
		pi := 1.0 / (1.0 + math.Exp(float64(-p.Ax[sample_index]*float32(p.Labels[sample_index]))))
		g_j = g_j + (-float32(p.Labels[sample_index]) * (1.0 - float32(pi)) * p.A_cols[j].Values[i])
		H_jj = H_jj + float32(pi)*(1.0-float32(pi))*p.A_cols[j].Values[i]*p.A_cols[j].Values[i]
		//		fmt.Printf("Ax(i): %f, pi: %f\n", p.Ax[sample_index], pi)
	}
	g_j = g_j / float32(p.L)
	H_jj = H_jj / float32(p.L)
	return g_j, H_jj
}
