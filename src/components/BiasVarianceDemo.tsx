import React, { useState, useEffect } from 'react';
import { Slider } from '../components/Slider';

interface DataPoint {
  x: number;
  y: number;
}

export default function BiasVarianceDemo() {
  const [complexity, setComplexity] = useState(3);
  const [showBias, setShowBias] = useState(true);
  const [showVariance, setShowVariance] = useState(true);
  const [showTotalError, setShowTotalError] = useState(true);

  // Generate synthetic data for demonstration
  const generateData = (): DataPoint[] => {
    const points: DataPoint[] = [];
    for (let i = 0; i <= 20; i++) {
      const x = i;
      // True function: quadratic with some noise
      const trueY = 0.05 * x * x - x + 10 + Math.sin(x * 0.5) * 2;
      points.push({ x, y: trueY });
    }
    return points;
  };

  const [trueData] = useState(generateData());

  // Calculate bias, variance, and total error based on complexity
  const calculateErrors = (complexity: number) => {
    const complexities = Array.from({ length: 10 }, (_, i) => i + 1);

    return complexities.map(c => {
      // Bias decreases as complexity increases (until overfitting)
      let bias = c <= 5 ? Math.max(0.1, 8 - c * 1.5) : Math.max(0.1, (c - 5) * 0.3 + 0.5);

      // Variance increases as complexity increases
      let variance = c <= 3 ? c * 0.3 : Math.pow(c - 2, 1.8) * 0.5;

      // Irreducible error (constant)
      const irreducibleError = 0.5;

      // Total error = bias² + variance + irreducible error
      const totalError = bias * bias + variance + irreducibleError;

      return {
        complexity: c,
        bias: bias * bias, // Show bias²
        variance,
        totalError,
        irreducibleError
      };
    });
  };

  const errorData = calculateErrors(complexity);
  const currentError = errorData.find(d => d.complexity === complexity);

  const maxError = Math.max(...errorData.map(d => Math.max(d.bias, d.variance, d.totalError)));

  return (
    <div className="bg-white dark:bg-gray-800 p-6 rounded-lg border border-gray-200 dark:border-gray-700">
      <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
        Interactive Bias-Variance Tradeoff Demo
      </h3>

      {/* Controls */}
      <div className="mb-6 space-y-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            Model Complexity: {complexity}
          </label>
          <Slider
            min={1}
            max={10}
            value={complexity}
            onChange={setComplexity}
            className="w-full"
          />
          <div className="flex justify-between text-xs text-gray-500 dark:text-gray-400 mt-1">
            <span>Simple (High Bias)</span>
            <span>Complex (High Variance)</span>
          </div>
        </div>

        <div className="flex space-x-4">
          <label className="flex items-center">
            <input
              type="checkbox"
              checked={showBias}
              onChange={(e) => setShowBias(e.target.checked)}
              className="mr-2"
            />
            <span className="text-sm text-red-600">Bias²</span>
          </label>
          <label className="flex items-center">
            <input
              type="checkbox"
              checked={showVariance}
              onChange={(e) => setShowVariance(e.target.checked)}
              className="mr-2"
            />
            <span className="text-sm text-blue-600">Variance</span>
          </label>
          <label className="flex items-center">
            <input
              type="checkbox"
              checked={showTotalError}
              onChange={(e) => setShowTotalError(e.target.checked)}
              className="mr-2"
            />
            <span className="text-sm text-green-600">Total Error</span>
          </label>
        </div>
      </div>

      {/* Visualization */}
      <div className="bg-gray-50 dark:bg-gray-900 p-4 rounded-lg">
        <svg width="100%" height="300" viewBox="0 0 400 300" className="overflow-visible">
          {/* Grid lines */}
          {Array.from({ length: 11 }, (_, i) => (
            <g key={i}>
              <line
                x1={40 + i * 32}
                y1={40}
                x2={40 + i * 32}
                y2={260}
                stroke="#e5e7eb"
                strokeWidth="1"
                opacity="0.3"
              />
              <line
                x1={40}
                y1={40 + i * 22}
                x2={360}
                y2={40 + i * 22}
                stroke="#e5e7eb"
                strokeWidth="1"
                opacity="0.3"
              />
            </g>
          ))}

          {/* Axes */}
          <line x1={40} y1={260} x2={360} y2={260} stroke="#374151" strokeWidth="2" />
          <line x1={40} y1={40} x2={40} y2={260} stroke="#374151" strokeWidth="2" />

          {/* Axis labels */}
          <text x={200} y={285} textAnchor="middle" className="text-xs fill-gray-600 dark:fill-gray-400">
            Model Complexity
          </text>
          <text x={20} y={150} textAnchor="middle" className="text-xs fill-gray-600 dark:fill-gray-400" transform="rotate(-90 20 150)">
            Error
          </text>

          {/* Data lines */}
          {showBias && (
            <polyline
              points={errorData.map((d, i) => `${40 + i * 32},${260 - (d.bias / maxError) * 220}`).join(' ')}
              fill="none"
              stroke="#dc2626"
              strokeWidth="3"
            />
          )}

          {showVariance && (
            <polyline
              points={errorData.map((d, i) => `${40 + i * 32},${260 - (d.variance / maxError) * 220}`).join(' ')}
              fill="none"
              stroke="#2563eb"
              strokeWidth="3"
            />
          )}

          {showTotalError && (
            <polyline
              points={errorData.map((d, i) => `${40 + i * 32},${260 - (d.totalError / maxError) * 220}`).join(' ')}
              fill="none"
              stroke="#16a34a"
              strokeWidth="3"
            />
          )}

          {/* Current complexity indicator */}
          <line
            x1={40 + (complexity - 1) * 32}
            y1={40}
            x2={40 + (complexity - 1) * 32}
            y2={260}
            stroke="#f59e0b"
            strokeWidth="2"
            strokeDasharray="5,5"
          />

          {/* X-axis labels */}
          {errorData.map((d, i) => (
            <text
              key={i}
              x={40 + i * 32}
              y={275}
              textAnchor="middle"
              className="text-xs fill-gray-600 dark:fill-gray-400"
            >
              {d.complexity}
            </text>
          ))}
        </svg>
      </div>

      {/* Current values */}
      {currentError && (
        <div className="mt-4 grid grid-cols-3 gap-4 text-center">
          <div className="p-3 bg-red-50 dark:bg-red-900/20 rounded-lg">
            <div className="text-lg font-semibold text-red-600 dark:text-red-400">
              {currentError.bias.toFixed(2)}
            </div>
            <div className="text-sm text-red-700 dark:text-red-300">Bias²</div>
          </div>
          <div className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
            <div className="text-lg font-semibold text-blue-600 dark:text-blue-400">
              {currentError.variance.toFixed(2)}
            </div>
            <div className="text-sm text-blue-700 dark:text-blue-300">Variance</div>
          </div>
          <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded-lg">
            <div className="text-lg font-semibold text-green-600 dark:text-green-400">
              {currentError.totalError.toFixed(2)}
            </div>
            <div className="text-sm text-green-700 dark:text-green-300">Total Error</div>
          </div>
        </div>
      )}

      {/* Explanation */}
      <div className="mt-4 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
        <p className="text-sm text-blue-800 dark:text-blue-200">
          <strong>Interactive Demo:</strong> Adjust the model complexity slider to see how bias and variance change.
          Simple models (low complexity) have high bias but low variance. Complex models have low bias but high variance.
          The optimal complexity minimizes the total error.
        </p>
      </div>
    </div>
  );
}