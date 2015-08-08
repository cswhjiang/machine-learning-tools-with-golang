package mathOp

import (
	"math"
)

func Abs(a float32) float32 {
	if a > 0 {
		return a
	} else {
		return -a
	}
}

func Exp(a float32) float32 {
	return float32(math.Exp(float64(a)))
}

func Log(a float32) float32 {
	return float32(math.Log(float64(a)))
}

func Sqrt(a float32) float32 {
	return float32(math.Sqrt(float64(a)))
}
