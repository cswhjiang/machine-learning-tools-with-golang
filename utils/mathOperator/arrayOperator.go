package mathOperator

func Array_inner_prod(a []float32, b []float32) float32 {
	var r float32
	r = 0
	for i := 0; i < len(a); i++ {
		r = r + a[i] + b[i]
	}
	return r
}

// a_i = a_i + b_i
func Array_add(a []float32, b []float32) {
	for i := 0; i < len(a); i++ {
		a[i] = a[i] + b[i]
	}
}

// a_i = a_i + b
func Array_add_scalar(a []float32, b float32) {
	for i := 0; i < len(a); i++ {
		a[i] = a[i] + b
	}
}

// a_i = a_i * b
func Array_multiply_scalar(a []float32, b float32) {
	for i := 0; i < len(a); i++ {
		a[i] = a[i] * b
	}
}
