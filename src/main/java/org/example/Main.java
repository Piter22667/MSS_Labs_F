package org.example;

import ij.IJ;
import ij.ImagePlus;
import ij.process.ImageProcessor;
import org.apache.commons.math3.linear.*;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

public class Main {
    public static void main(String[] args) throws IOException {
        // Read images
        ImagePlus xImage = IJ.openImage("x3.bmp");
        ImagePlus yImage = IJ.openImage("y8.bmp");

        // Convert images to matrices
        RealMatrix X = imageToMatrix(xImage);
        RealMatrix Y = imageToMatrix(yImage);

        System.out.println("X dimensions: " + X.getRowDimension() + "x" + X.getColumnDimension());
        System.out.println("Y dimensions: " + Y.getRowDimension() + "x" + Y.getColumnDimension());

        // Add row of ones to X
        X = addRowOfOnes(X);

        System.out.println("X dimensions after adding row of ones: " + X.getRowDimension() + "x" + X.getColumnDimension());

        // Calculate pseudoinverse using Greville formula
        RealMatrix XPseudoInverse = calculateGrevillePseudoinverse(X);

        System.out.println("XPseudoInverse dimensions: " + XPseudoInverse.getRowDimension() + "x" + XPseudoInverse.getColumnDimension());

        // Calculate transformation matrix A
        RealMatrix Z = MatrixUtils.createRealIdentityMatrix(X.getRowDimension()).subtract(X.multiply(XPseudoInverse));
        RealMatrix V = MatrixUtils.createRealMatrix(Y.getRowDimension(), X.getRowDimension());
        RealMatrix A = Y.multiply(XPseudoInverse).add(V.multiply(Z));

        System.out.println("A dimensions: " + A.getRowDimension() + "x" + A.getColumnDimension());

        // Apply transformation
        RealMatrix YTransformed = A.multiply(X);

        System.out.println("YTransformed dimensions: " + YTransformed.getRowDimension() + "x" + YTransformed.getColumnDimension());

        // Project values to [0, 255] range
        RealMatrix YProjected = projectTo255Range(YTransformed);

        // Save the result
        saveMatrixAsImage(YProjected, "resultTransformed.bmp");

        // Calculate and print mean squared error
        double mse = calculateMeanSquaredError(Y, YProjected);
        System.out.println("Mean Squared Error: " + mse);
    }

    private static RealMatrix imageToMatrix(ImagePlus image) {
        ImageProcessor ip = image.getProcessor();
        int width = ip.getWidth();
        int height = ip.getHeight();
        double[][] data = new double[height][width];
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                data[y][x] = ip.getPixelValue(x, y);
            }
        }
        return MatrixUtils.createRealMatrix(data);
    }

    private static RealMatrix addRowOfOnes(RealMatrix matrix) {
        int rows = matrix.getRowDimension();
        int cols = matrix.getColumnDimension();
        double[][] newData = new double[rows + 1][cols];
        for (int i = 0; i < rows; i++) {
            System.arraycopy(matrix.getRow(i), 0, newData[i], 0, cols);
        }
        for (int j = 0; j < cols; j++) {
            newData[rows][j] = 1.0;
        }
        return MatrixUtils.createRealMatrix(newData);
    }

    private static RealMatrix calculateGrevillePseudoinverse(RealMatrix X) {
        int m = X.getRowDimension();
        int n = X.getColumnDimension();
        RealMatrix psinv = MatrixUtils.createRealMatrix(n, m);
        double epsilon = 0.0001;

        for (int i = 0; i < m; i++) {
            RealVector xi = X.getRowVector(i);

            if (i == 0) {
                double dotpr = xi.dotProduct(xi);
                if (Math.abs(dotpr) < epsilon) {
                    psinv.setColumnVector(i, new ArrayRealVector(n));
                } else {
                    psinv.setColumnVector(i, xi.mapDivide(dotpr));
                }
            } else {
                RealMatrix XCur = X.getSubMatrix(0, i-1, 0, n-1);
                RealMatrix Z = MatrixUtils.createRealIdentityMatrix(n).subtract(psinv.getSubMatrix(0, n-1, 0, i-1).multiply(XCur));
                double denom = xi.dotProduct(Z.operate(xi));
                RealVector numer;
                if (Math.abs(denom) < epsilon) {
                    RealMatrix R = psinv.getSubMatrix(0, n-1, 0, i-1).multiply(psinv.getSubMatrix(0, n-1, 0, i-1).transpose());
                    numer = R.operate(xi);
                    denom = 1 + xi.dotProduct(R.operate(xi));
                } else {
                    numer = Z.operate(xi);
                }
                RealMatrix psinvUpdate = numer.outerProduct(xi).multiply(psinv.getSubMatrix(0, n-1, 0, i-1)).scalarMultiply(1/denom);
                psinv.setSubMatrix(psinv.getSubMatrix(0, n-1, 0, i-1).subtract(psinvUpdate).getData(), 0, 0);
                psinv.setColumnVector(i, numer.mapDivide(denom));
            }
        }
        return psinv;
    }

    private static RealMatrix projectTo255Range(RealMatrix matrix) {
        double min = getMinValue(matrix);
        double max = getMaxValue(matrix);
        return matrix.scalarAdd(-min).scalarMultiply(255 / (max - min));
    }

    private static double getMinValue(RealMatrix matrix) {
        double min = Double.MAX_VALUE;
        for (int i = 0; i < matrix.getRowDimension(); i++) {
            for (int j = 0; j < matrix.getColumnDimension(); j++) {
                min = Math.min(min, matrix.getEntry(i, j));
            }
        }
        return min;
    }

    private static double getMaxValue(RealMatrix matrix) {
        double max = Double.MIN_VALUE;
        for (int i = 0; i < matrix.getRowDimension(); i++) {
            for (int j = 0; j < matrix.getColumnDimension(); j++) {
                max = Math.max(max, matrix.getEntry(i, j));
            }
        }
        return max;
    }

    private static void saveMatrixAsImage(RealMatrix matrix, String filename) throws IOException {
        int height = matrix.getRowDimension();
        int width = matrix.getColumnDimension();
        BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int value = (int) Math.round(matrix.getEntry(y, x));
                int rgb = (value << 16) | (value << 8) | value;
                image.setRGB(x, y, rgb);
            }
        }
        ImageIO.write(image, "PNG", new File(filename));
    }

    private static double calculateMeanSquaredError(RealMatrix original, RealMatrix projected) {
        if (original.getRowDimension() != projected.getRowDimension() ||
                original.getColumnDimension() != projected.getColumnDimension()) {
            throw new IllegalArgumentException("Matrices must have the same dimensions");
        }

        double sumSquaredDiff = 0;
        int totalElements = original.getRowDimension() * original.getColumnDimension();

        for (int i = 0; i < original.getRowDimension(); i++) {
            for (int j = 0; j < original.getColumnDimension(); j++) {
                double diff = original.getEntry(i, j) - projected.getEntry(i, j);
                sumSquaredDiff += diff * diff;
            }
        }

        return sumSquaredDiff / totalElements;
    }

}