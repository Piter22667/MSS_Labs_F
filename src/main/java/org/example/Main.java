package org.example;

import org.apache.commons.math3.linear.*;
import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
//import org.jfree.chart.ChartFactory;
////import org.jfree.chart.ChartUtils;
//import org.jfree.chart.JFreeChart;
//import org.jfree.data.xy.XYSeries;
//import org.jfree.data.xy.XYSeriesCollection;

public class Main {
    public static void main(String[] args) {
        try {

            // Зчитування файлів
            double[][] X = readImageToMatrix("x1.bmp");
            double[][] Y = readImageToMatrix("y8.bmp");

            System.out.println("Розмірність X: " + X.length + "x" + X[0].length);
            System.out.println("Розмірність Y: " + Y.length + "x" + Y[0].length);

            X = addOnesRow(X);
            RealMatrix matrixX = MatrixUtils.createRealMatrix(X);
            RealMatrix matrixY = MatrixUtils.createRealMatrix(Y);

            long startTime = System.nanoTime();
            Runtime runtime = Runtime.getRuntime();
            long usedMemoryBefore = runtime.totalMemory() - runtime.freeMemory();

            RealMatrix pseudoInverse = findPseudoInverse(matrixX);
            System.out.println("Розмірність псевдооберненої матриці X+: " +
                    pseudoInverse.getRowDimension() + "x" +
                    pseudoInverse.getColumnDimension());

            long endTime = System.nanoTime();
            long usedMemoryAfter = runtime.totalMemory() - runtime.freeMemory();

            long duration = (endTime - startTime) / 1_000_000;
            long memoryUsed = usedMemoryAfter - usedMemoryBefore;

            System.out.println("\nЗагальний час виконання: " + duration + " мс");
            System.out.println("Загальне використання пам'яті: " + memoryUsed / 1024 + " КБ\n");





            // Перевірка характеристичної властивості псевдооберненої матриці
            RealMatrix X_plus_X_X_plus = pseudoInverse.subtract(pseudoInverse.multiply(matrixX).multiply(pseudoInverse));
            double mse1 = calculateMSE(X_plus_X_X_plus);
            System.out.println("Середньоквадратичне відхилення (X+)X(X+): " + mse1);

            RealMatrix X_X_plus_X = matrixX.subtract(matrixX.multiply(pseudoInverse).multiply(matrixX));
            double mse2 = calculateMSE(X_X_plus_X);
            System.out.println("Середньоквадратичне відхилення X(X+)X від X: " + mse2);


            // Перевірка властивостей псевдооберненої матриці
            checkPseudoInverseProperties(matrixX, pseudoInverse);

            // Шукаємо оператор А перетворення X в Y
            int m = matrixX.getRowDimension();
            int p = matrixY.getRowDimension();
            RealMatrix Z = MatrixUtils.createRealIdentityMatrix(m).subtract(matrixX.multiply(pseudoInverse));
            RealMatrix V = MatrixUtils.createRealMatrix(p, m);
            RealMatrix A_MP = matrixY.multiply(pseudoInverse).add(V.multiply(Z));

            // Застосовуємо оператор A до X
            RealMatrix Yimage_MP = A_MP.multiply(matrixX);

            // Проекція елементів матриці на проміжок [0; 255]
            RealMatrix Yimage_projected_MP = projectTo0_255(Yimage_MP);

            // Зберігаємо результат як зображення
            saveMatrixAsImage(Yimage_projected_MP, "Result_image.bmp");

            // Середньоквадратична похибка знаходження образу
            double mse = calculateMSE(matrixY.subtract(Yimage_projected_MP));
            System.out.println("Середньоквадратична похибка знаходження образу: " + mse);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }


    private static double findMinValue(RealMatrix matrix) {
        double min = Double.MAX_VALUE;
        for (int i = 0; i < matrix.getRowDimension(); i++) {
            for (int j = 0; j < matrix.getColumnDimension(); j++) {
                min = Math.min(min, matrix.getEntry(i, j));
            }
        }
        return min;
    }

    private static double findMaxValue(RealMatrix matrix) {
        double max = Double.MIN_VALUE;
        for (int i = 0; i < matrix.getRowDimension(); i++) {
            for (int j = 0; j < matrix.getColumnDimension(); j++) {
                max = Math.max(max, matrix.getEntry(i, j));
            }
        }
        return max;
    }


    private static RealMatrix findPseudoInverse(RealMatrix X) {
        long startTime = System.nanoTime();
        Runtime runtime = Runtime.getRuntime();
        long usedMemoryBefore = runtime.totalMemory() - runtime.freeMemory();

        double epsilon = 1e-10;
        double delta = 10.0;
        int m = X.getRowDimension();
        int n = X.getColumnDimension();

        RealMatrix XT = X.transpose();
        RealMatrix XXT = X.multiply(XT);
        RealMatrix I = MatrixUtils.createRealIdentityMatrix(m);

        RealMatrix pseudoInv_prev = XT.multiply(new QRDecomposition(XXT.add(I.scalarMultiply(delta))).getSolver().getInverse());

        int maxIterations = 100;
        System.out.println("\nІтерації методу Мура-Пенроуза:");
        for (int i = 0; i < maxIterations; i++) {
            delta /= 2;
            try {
                RealMatrix pseudoInv_cur = XT.multiply(new QRDecomposition(XXT.add(I.scalarMultiply(delta))).getSolver().getInverse());

                double diff = calculateNorm(pseudoInv_cur.subtract(pseudoInv_prev));
                System.out.printf("Ітерація %d: delta = %.10f, різниця = %.10f%n", i+1, delta, diff);

                if (diff < epsilon || delta < 1e-15) {
                    System.out.println("Збіжність досягнута після " + (i+1) + " ітерацій");
                    System.out.printf("Оптимальне значення delta: %.10f%n", delta);
                    return pseudoInv_cur;
                }
                pseudoInv_prev = pseudoInv_cur;
            } catch (SingularMatrixException e) {
                System.out.println("Зустрінута сингулярна матриця на ітерації " + i + ". Повертаємо попередній результат.");
                System.out.printf("Оптимальне значення delta: %.10f%n", delta * 2);  // Повертаємо попереднє значення delta
                return pseudoInv_prev;
            }
        }

        System.out.println("Попередження: Досягнуто максимальну кількість ітерацій у findPseudoInverse");
        System.out.printf("Оптимальне значення delta: %.10f%n", delta);

        long usedMemoryAfter = runtime.totalMemory() - runtime.freeMemory();
        long memoryUsed = usedMemoryAfter - usedMemoryBefore;
        long endTime = System.nanoTime();
        long duration = (endTime - startTime) / 1_000_000;

        return pseudoInv_prev;
    }




    private static void checkPseudoInverseProperties(RealMatrix A, RealMatrix A_plus) {
        double epsilon = 1e-6; // Допустима відносна похибка

        System.out.println("\n\nВластивості псевдооберненої матриці: ");
        RealMatrix AA_plus_A = A.multiply(A_plus).multiply(A);
        double relError1 = calculateRelativeError(AA_plus_A, A);
//        System.out.println("Відносна похибка (AA+A - A)/A: " + relError1);
        System.out.println("Властивість AA+A = A: " + (relError1 < epsilon));

        // Властивість 2: A+AA+ = A+
        RealMatrix A_plus_A_A_plus = A_plus.multiply(A).multiply(A_plus);
        double relError2 = calculateRelativeError(A_plus_A_A_plus, A_plus);
//        System.out.println("Відносна похибка (A+AA+ - A+)/A+: " + relError2);
        System.out.println("Властивість A+AA+ = A+: " + (relError2 < epsilon));

        // Властивість 3: AA+ - симетрична матриця
        RealMatrix AA_plus = A.multiply(A_plus);
        double relError3 = calculateRelativeError(AA_plus, AA_plus.transpose());
//        System.out.println("Відносна похибка (AA+ - (AA+)^T)/AA+: " + relError3);
        System.out.println("Властивість AA+ - симетрична: " + (relError3 < epsilon));

        // Властивість 4: A+A - симетрична матриця
        RealMatrix A_plus_A = A_plus.multiply(A);
        double relError4 = calculateRelativeError(A_plus_A, A_plus_A.transpose());
//        System.out.println("Відносна похибка (A+A - (A+A)^T)/A+A: " + relError4);
        System.out.println("Властивість A+A - симетрична: " + (relError4 < epsilon) + "\n");
    }

    private static double calculateRelativeError(RealMatrix A, RealMatrix B) {
        return calculateNorm(A.subtract(B)) / calculateNorm(B);
    }


    private static double calculateNorm(RealMatrix matrix) {
        double maxRowSum = 0;
        for (int i = 0; i < matrix.getRowDimension(); i++) {
            double rowSum = 0;
            for (int j = 0; j < matrix.getColumnDimension(); j++) {
                rowSum += Math.abs(matrix.getEntry(i, j));
            }
            maxRowSum = Math.max(maxRowSum, rowSum);
        }
        return maxRowSum;
    }

    private static double calculateMSE(RealMatrix matrix) {
        double sum = 0;
        int elements = matrix.getRowDimension() * matrix.getColumnDimension();
        for (int i = 0; i < matrix.getRowDimension(); i++) {
            for (int j = 0; j < matrix.getColumnDimension(); j++) {
                sum += Math.pow(matrix.getEntry(i, j), 2);
            }
        }
        return sum / elements;
    }

    private static RealMatrix projectTo0_255(RealMatrix matrix) {
        double min = findMinValue(matrix);
        double max = findMaxValue(matrix);
        return matrix.scalarAdd(-min).scalarMultiply(255 / (max - min));
    }

    private static void saveMatrixAsImage(RealMatrix matrix, String filename) throws IOException {


        int width = matrix.getColumnDimension();
        int height = matrix.getRowDimension();
        BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int gray = (int) Math.round(matrix.getEntry(y, x));
                gray = Math.max(0, Math.min(255, gray));
                int rgb = (gray << 16) | (gray << 8) | gray;
                image.setRGB(x, y, rgb);
            }
        }
        System.out.println("Збереження зображення: " + filename);

        ImageIO.write(image, "bmp", new File("Result_image.bmp"));
    }

    private static double[][] addOnesRow(double[][] matrix) {
        int rows = matrix.length;
        int cols = matrix[0].length;
        double[][] newMatrix = new double[rows + 1][cols];

        for (int i = 0; i < rows; i++) {
            System.arraycopy(matrix[i], 0, newMatrix[i], 0, cols);
        }
        for (int j = 0; j < cols; j++) {
            newMatrix[rows][j] = 1.0;
        }

        return newMatrix;
    }

    private static double[][] readImageToMatrix(String filename) throws IOException {
        BufferedImage image = ImageIO.read(new File(filename));
        int width = image.getWidth();
        int height = image.getHeight();
        double[][] matrix = new double[height][width];

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int rgb = image.getRGB(x, y);
                int gray = (rgb >> 16) & 0xFF; // Припускаємо, що зображення вже в відтінках сірого
                matrix[y][x] = gray;
            }
        }

        return matrix;
    }


    private static double[][] normalizeMatrix(double[][] matrix) {
        double min = Double.MAX_VALUE;
        double max = Double.MIN_VALUE;
        for (double[] row : matrix) {
            for (double value : row) {
                min = Math.min(min, value);
                max = Math.max(max, value);
            }
        }
        double range = max - min;
        double[][] normalized = new double[matrix.length][matrix[0].length];
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[i].length; j++) {
                normalized[i][j] = (matrix[i][j] - min) / range;
            }
        }
        return normalized;
    }


}