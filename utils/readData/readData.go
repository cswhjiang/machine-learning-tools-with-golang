// read libsvm format data
package readData

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"strings"
)

//type FeatureNode struct {
//	index int
//	value float32
//}
//no use for now

func ReadData(fileName string, isClassification bool) (*Problem, error) { //faster than ReadData2
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

	var row_element_array []int        //numer of element in each row
	var col_element_array []int        //numer of element in each column
	col_count_map := make(map[int]int) //for  col_element_array
	for scanner.Scan() {
		line := scanner.Text()
		num_sample++
		nodes := strings.Split(line, " ")
		num_nodes = num_nodes + len(nodes)
		row_element_array = append(row_element_array, len(nodes)-1)
		for i := 1; i < len(nodes); i++ {
			node_str_array := strings.Split(nodes[i], ":")
			feature_index_str := node_str_array[0]

			feature_index, _ := strconv.Atoi(feature_index_str)
			count_temp := col_count_map[feature_index-1]
			col_count_map[feature_index-1] = count_temp + 1
		}

		last_str := nodes[len(nodes)-1]
		str_array := strings.Split(last_str, ":")
		feature_index, _ := strconv.Atoi(str_array[0])
		if feature_index > num_feature {
			num_feature = feature_index
		}
	}

	// fill col_element_array
	col_element_array = make([]int, num_feature)
	for key, value := range col_count_map {
		col_element_array[key] = value
	}

	//	fmt.Printf("samples: %d, featuers: %d ,  nodes: %d\n", num_sample, num_feature, num_nodes)
	if err := scanner.Err(); err != nil {
		fmt.Fprintln(os.Stderr, "reading from file:", err)
	}

	prob := new(Problem)
	prob.reserve(num_sample, num_feature, isClassification, row_element_array, col_element_array) //reserve space

	row_position_array := make([]int, len(row_element_array))
	col_position_array := make([]int, len(col_element_array))
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
			row_position := row_position_array[sample_index]
			col_position := col_position_array[feature_index-1]
			prob.addNode(label_current, sample_index, feature_index-1, float32(value), row_position, col_position)
			row_position_array[sample_index] = row_position + 1
			col_position_array[feature_index-1] = col_position + 1
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
