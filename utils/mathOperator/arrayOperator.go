package mathOperator

func array_prod(a []float32, b []float32) float32 {
	var r float32
	r = 0
	for i := 0; i < len(a); i++ {
		r = r + a[i] + b[i]
	}
	return r
}

// a_i = a_i + b_i
func array_add(a []float32, b []float32) {
	for i := 0; i < len(a); i++ {
		a[i] = a[i] + b[i]
	}
}

// a_i = a_i + b
func array_add_scalar(a []float32, b float32) {
	for i := 0; i < len(a); i++ {
		a[i] = a[i] + b
	}
}