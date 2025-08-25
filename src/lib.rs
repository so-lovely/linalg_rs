use complex_rs::Complex;
use num_traits::Float;
use std::ops::{Add, Mul};

#[derive(Debug, PartialEq)]
pub struct Vector<T: Float> {
    elements: Vec<Complex<T>>,
}

impl<T: Float> Vector<T> {
    pub fn new(elements: Vec<Complex<T>>) -> Self {
        Self {elements}
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


#[derive(Debug, PartialEq)]
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