package readData

import (
	"fmt"
	"os"
)

type DenseVector struct {
	Values []float32
}

func (d *DenseVector) Reserve(l int) {
	d.Values = make([]float32, l)
}
func (d *DenseVector) Add_scalar(a float32) {
	for i := 0; i < len(d.Values); i++ {
		d.Values[i] = d.Values[i] + a
	}
}

func (d *DenseVector) Multiply_scalar(a float32) {
	for i := 0; i < len(d.Values); i++ {
		d.Values[i] = d.Values[i] * a
	}
}

//r = v^T a
func (d *DenseVector) Multiply_dense_vector(a *DenseVector) float32 {
	if len(d.Values) != len(a.Values) {
		os.Exit(1)
		//		return -1
	}
	var r float32
	r = 0

	for i := 0; i < len(d.Values); i++ {
		r = r + d.Values[i]*a.Values[i]
	}
	return r
}

//r = v^T a
func (d *DenseVector) Multiply_sparse_vector(a *SparseVector) float32 {
	var r float32
	r = 0

	for i := 0; i < len(a.Values); i++ {
		r = r + d.Values[i]*a.Values[i]
	}
	return r
}

//sparse vector
type SparseVector struct {
	Idxs   []int //index are sorted
	Values []float32
	//	nnz    int //number of non-zero elements, no use for now
	//	dim    int //dimensionality or length, no use for now
}

//func (s *SparseVector) length() int {
//	return s.dim
//}

func (sv *SparseVector) add_element(index int, value float32) {
	sv.Idxs = append(sv.Idxs, index)
	sv.Values = append(sv.Values, value)
	//	sv.nnz = sv.nnz + 1
}

// v =  v+a
func (v *SparseVector) Add_scalar(a float32) {
	for i := 0; i < len(v.Values); i++ {
		v.Values[i] = v.Values[i] + a
	}
}

// v = v.*a
func (v *SparseVector) Multiply_scalar(a float32) {
	for i := 0; i < len(v.Values); i++ {
		v.Values[i] = v.Values[i] * a
	}
}

//r = v^T a
func (v *SparseVector) Multiply_sparse_vector(a *SparseVector) float32 {
	var r float32
	r = 0
	i := 0
	j := 0
	for i < len(v.Values) && j < len(a.Values) {
		if v.Idxs[i] == a.Idxs[j] {
			r = r + v.Values[i]*a.Values[j]
			i++
			j++
		} else if v.Idxs[i] > a.Idxs[j] {
			j++
		} else {
			i++
		}
	}
	return r
}

//r = v^T a
func (v *SparseVector) Multiply_dense_array(a []float32) float32 {
	var r float32
	r = 0

	for i := 0; i < len(v.Values); i++ {
		r = r + v.Values[i]*a[v.Idxs[i]]
	}
	return r
}

func (sv *SparseVector) getString() string {
	var str string
	for i := 0; i < len(sv.Idxs); i++ {
		str += fmt.Sprintf("%d:%f ", sv.Idxs[i], sv.Values[i])
	}
	return str
}

func (sv *SparseVector) String() string {
	var str string
	for i := 0; i < len(sv.Idxs); i++ {
		str += fmt.Sprintf("%d:%f ", sv.Idxs[i], sv.Values[i])
	}
	return str
}

type Problem struct {
	L      int            // number of samples
	N      int            //number of feature
	Labels []int          //label
	A_rows []SparseVector //rows
	A_cols []SparseVector //columns. Redundant.

	X []float32 //parameter
	//	b    float32 // do not consider bias term for now
	Size int // number of nodes(non-zero elements)

	Max_iter int
	Lambda   float32
	Epsilon  float32

	IsClassification bool // true if it is a classification problem
	NumClass         int  //only defined if isClassification == true
}

func (p *Problem) reserve(num_sample int, num_feature int, isClassification bool) {
	p.L = num_sample
	p.N = num_feature
	p.Size = 0
	p.A_cols = make([]SparseVector, num_feature)
	p.A_rows = make([]SparseVector, num_sample)
	p.X = make([]float32, num_feature)
	p.Labels = make([]int, num_sample)
	p.Max_iter = 100
	p.Lambda = 0.0000001
	p.Epsilon = 0.001
	p.IsClassification = isClassification
	p.NumClass = 2 //we consider binary classifition by default
}

func (p *Problem) addNode(y int, sample_index int, feature_index int, value float32) {
	//	fmt.Printf("add node: %d %d %f\n", sample_index, feature_index, value)
	p.Labels[sample_index] = y
	p.A_cols[feature_index].add_element(sample_index, value)
	p.A_rows[sample_index].add_element(feature_index, value)
	p.Size = p.Size + 1
}

func (p *Problem) PrintProblem() {
	fmt.Printf("samples: %d, featuers: %d \n", p.L, p.N)
	fmt.Printf("number of non-zeros elements: %d \n", p.Size)
	fmt.Printf("maxmum iterations: %d \n", p.Max_iter)
	fmt.Printf("lambda: %e, epsilon %e\n", p.Lambda, p.Epsilon)
	//	for i := 0; i < p.L; i++ {
	//		fmt.Printf("%d %s\n", p.Labels[i], p.A_rows[i].getString())
	//	}
	//	for i := 0; i < p.n; i++ {
	//		fmt.Printf(" %s\n", p.A_cols[i].getString())
	//	}
}