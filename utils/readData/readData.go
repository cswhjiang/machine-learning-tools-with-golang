// read libsvm format data
package readData

import (
	"bufio"
	"fmt"
	//	"io"
	"os"
	"strconv"
	"strings"
)

//type FeatureNode struct {
//	index int
//	value float32
//}
//no use for now
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
}

func (p *Problem) reserve(num_sample int, num_feature int) {
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

func ReadData(fileName string) (*Problem, error) { //faster than ReadData2
	f, err := os.Open(fileName)
	if err != nil {
		fmt.Println(err)
		return nil, err
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)

	num_sample := 0  //number of samples
	num_feature := 0 //number of features
	num_nodes := 0   //number of nodes
	for scanner.Scan() {
		line := scanner.Text()
		num_sample++
		nodes := strings.Split(line, " ")
		num_nodes = num_nodes + len(nodes)
		last_str := nodes[len(nodes)-1]
		str_array := strings.Split(last_str, ":")
		feature_index, _ := strconv.Atoi(str_array[0])
		if feature_index > num_feature {
			num_feature = feature_index
		}
	}

	//	fmt.Printf("samples: %d, featuers: %d ,  nodes: %d\n", num_sample, num_feature, num_nodes)
	if err := scanner.Err(); err != nil {
		fmt.Fprintln(os.Stderr, "reading from file:", err)
	}

	prob := new(Problem)
	prob.reserve(num_sample, num_feature)

	_, err = f.Seek(0, 0)
	scanner = bufio.NewScanner(f)
	sample_index := 0

	min_feature_index := 100
	for scanner.Scan() {
		line := scanner.Text()
		//		fmt.Println(line)
		nodes := strings.Split(line, " ")

		label_current, _ := strconv.Atoi(nodes[0])
		for i := 1; i < len(nodes); i++ {
			node_str_array := strings.Split(nodes[i], ":")
			feature_index_str := node_str_array[0]
			value_str := node_str_array[1]

			feature_index, _ := strconv.Atoi(feature_index_str)
			if feature_index < min_feature_index {
				min_feature_index = feature_index
			}
			value, _ := strconv.ParseFloat(value_str, 32)
			prob.addNode(label_current, sample_index, feature_index-1, float32(value))
		}

		sample_index++
	}
	//	fmt.Printf("min feature index:%d \n", min_feature_index)
	return prob, nil
}

//func ReadData2(filename string) {
//	f, err := os.Open(filename)
//	if err != nil {
//		fmt.Println(err)
//		return
//	}
//	defer f.Close()
//	r := bufio.NewReader(f)
//	line, err := r.ReadString('\n')
//	l := 1
//	_ = line
//	for err == nil {
//		line, err = r.ReadString('\n')
//		_ = line
//		l++
//	}
//	fmt.Printf("lines: %d \n", l)
//	if err != io.EOF {
//		fmt.Println(err)
//		return
//	}

//}
