use complex_rs::Complex;
use num_traits::Float;
use std::ops::{Add, Mul};

#[derive(Debug, PartialEq, Clone)]
pub struct Vector<T: Float> {
    elements: Vec<Complex<T>>,
}

impl<T: Float> Vector<T> {
    pub fn new(elements: Vec<Complex<T>>) -> Self {
        Self {elements}
    }
    pub fn elements(&self) -> &[Complex<T>] {
        &self.elements
    }
    
    pub fn get(&self, index: usize) -> Complex<T> {
        self.elements[index]
    }
    pub fn dim(&self) -> usize {
        self.elements.len()
    }

    pub fn inner_product(&self, other: &Self) -> Complex<T> {
        assert_eq!(self.dim(), other.dim(), "Vectors must have the same dimension.");

        self.elements
            .iter()
            .zip(other.elements.iter())
            .map(|(a,b)| a.conjugate() * *b)
            .fold(Complex::zero(), |acc, val| acc+ val)
    }
}

impl<T: Float> Add for Vector<T> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        assert_eq!(self.dim(), rhs.dim(), "Vectors must have the same dimension");
        let elements = self.elements
            .into_iter()
            .zip(rhs.elements.into_iter())
            .map(|(a,b)| a+b)
            .collect();
        Self::new(elements)
    }
}

impl<T: Float> Mul<T> for Vector<T> {
    type Output = Self;
    fn mul(self, rhs: T) -> Self {
        let elements = self.elements.into_iter().map(|elem|
        elem * rhs).collect();
            Self::new(elements)
    }

}


#[derive(Debug, PartialEq, Clone)]
pub struct Matrix<T: Float> {
    rows: usize,
    cols: usize,
    elements: Vec<Complex<T>>,
}

impl<T: Float> Matrix<T> {
    pub fn new(rows: usize, cols: usize, elements:Vec<Complex<T>>) -> Self {
        assert_eq!(rows * cols, elements.len(), "Number of elements must match rows * cols.");
        Self { rows, cols, elements }
    }
    pub fn get(&self, row: usize, col: usize) -> Complex<T> {
        self.elements[row * self.cols + col]
    }

    pub fn tensor_product(&self, other:&Self) -> Self {
        let new_rows = self.rows * other.rows;
        let new_cols = self.cols * other.cols;
        let mut new_elements = vec![Complex::zero(); new_rows * new_cols];
        
        for i in 0..self.rows {
            for j in 0..self.cols {
                let self_elem = self.get(i,j);
                for k in 0..other.rows {
                    for l in 0..other.cols {
                        let other_elem = other.get(k,l);
                        let new_row = i * other.rows + k;
                        let new_col = j * other.cols + l;
                        new_elements[new_row * new_cols + new_col] = self_elem * other_elem;
                    }
                }
            }
        }
        Matrix::new(new_rows, new_cols, new_elements)

    }
    
    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn elements(&self) -> &[Complex<T>] {
        &self.elements
    }
}

impl<T: Float> Mul<Vector<T>> for Matrix<T> {
    type Output = Vector<T>;
    fn mul(self, rhs: Vector<T>) -> Vector<T> {
        assert_eq!(self.cols, rhs.dim(), "Matrix cols must match vector dimension.");
        let mut result_elements = Vec::with_capacity(self.rows);

        for i in 0..self.rows{
            let mut sum = Complex::zero();
            for j in 0..self.cols {
                sum = sum + self.get(i,j) * rhs.elements[j];
            }
            result_elements.push(sum);
        }
        Vector::new(result_elements)
    }

}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]

    fn test_tensor_product() {
        let a = Matrix::new(2,2,vec![
            Complex::new(1.0,0.0), Complex::new(2.0, 0.0),
            Complex::new(3.0,0.0), Complex::new(4.0, 0.0),
        ]);

        let b = Matrix::new(1,2, vec![
            Complex::new(0.0,0.0), Complex::new(5.0,0.0)
        ]);
        let result = a.tensor_product(&b);
        let expected = Matrix::new(2,4, vec![
            Complex::new(0.0,0.0), Complex::new(5.0,0.0), Complex::new(0.0,0.0), Complex::new(10.0,0.0),
            Complex::new(0.0,0.0), Complex::new(15.0,0.0), Complex::new(0.0,0.0), Complex::new(20.0,0.0),

        ]);
        assert_eq!(result, expected);
    }
    
    #[test]
    fn test_matrix_vector_multiplication() {
        let matrix_x: Matrix<f64> = Matrix::new(2,2, vec![
            Complex::zero(), Complex::one(),
            Complex::one(), Complex::zero(),
        ]);

        let state_zero = Vector::<f64>::new(vec![Complex::one(), Complex::zero()]);
        let result = matrix_x * state_zero;
        let expected = Vector::<f64>::new(vec![Complex::zero(), Complex::one()]);
        assert_eq!(result, expected);
    }
}