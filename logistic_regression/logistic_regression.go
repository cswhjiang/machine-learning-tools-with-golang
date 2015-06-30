// logistic_regression
package main

import (
	"fmt"
	"github.com/cswhjiang/machine-learning-tools-with-golang/utils/readData"
	"time"
)

type TestTruct struct {
	a int
}

func (t TestTruct) getA() int {
	return t.a
}
func main() {
	start := time.Now()

	p, _ := readData.ReadData("/home/jwh/dataset/rcv1/test.libsvm")
	//	p, _ := readData.ReadData("/home/jwh/rcv1/rcv1_test.binary")

	p.PrintProblem()
	_ = p
	fmt.Println("logistric regression!")

	elapsed := time.Since(start)
	fmt.Printf("took %s \n", elapsed)
}
