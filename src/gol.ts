/**
 * @license
 * Copyright 2017 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import {Array2D, Graph, NDArray, NDArrayMathGPU, Session, SGDOptimizer} from 'deeplearn';
import {InCPUMemoryShuffledInputProviderBuilder} from 'deeplearn/dist/data/input_provider';
import {Tensor} from 'deeplearn/dist/graph/graph';
import {CostReduction, FeedEntry} from 'deeplearn/dist/graph/session';
import {NDArrayMath} from 'deeplearn/dist/math/math';
import {Scalar} from 'deeplearn/dist/math/ndarray';
import {expectArrayInMeanStdRange} from 'deeplearn/dist/test_util';
import {Server} from 'http';
import { AdamaxOptimizer } from 'deeplearn/dist/graph/optimizers/adamax_optimizer';
import { AdagradOptimizer } from 'deeplearn/dist/graph/optimizers/adagrad_optimizer';

/* Test-only method for logging worlds. */
function testPrint(array: NDArray, size: number) {
  let t = [];
  let v = array.getValues();
  for (let i = 0; i < v.length; i++) {
    t.push(v[i]);
    if (t.length == size) {
      console.log(t);
      t = [];
    }
  }
  console.log('');
}

/**
 * Main class for running a deep-neural network of training for Game-of-life
 * next sequence.
 */
class GameOfLife {
  session: Session;
  math: NDArrayMath = new NDArrayMathGPU();
  batchSize = 1;

  // An optimizer with a certain initial learning rate. Used for training.
  initialLearningRate = 0.042;
  optimizer: SGDOptimizer;
  // optimizer: AdagradOptimizer;

  inputTensor: Tensor;
  targetTensor: Tensor;
  costTensor: Tensor;
  predictionTensor: Tensor;

  size: number;
  step = 0;

  // Maps tensors to InputProviders
  feedEntries: FeedEntry[];

  constructor(size: number) {
    this.size = size;
    // this.optimizer = new AdagradOptimizer(0.01);
    this.optimizer = new SGDOptimizer(this.initialLearningRate);
  }

  setupSession(): void {
    const graph = new Graph();
    const size = this.size * this.size;

    this.inputTensor = graph.placeholder('input', [size]);
    this.targetTensor = graph.placeholder('target', [size]);

    let hiddenLayer =
        GameOfLife.createFullyConnectedLayer(graph, this.inputTensor, 0, size);
    hiddenLayer =
        GameOfLife.createFullyConnectedLayer(graph, hiddenLayer, 1, size);
    this.predictionTensor =
        GameOfLife.createFullyConnectedLayer(graph, hiddenLayer, 2, size);
        // GameOfLife.createFullyConnectedLayerSigmoid(graph, hiddenLayer, 2, size);

    // This is wrong - need to use something that is not mean-squared...
    this.costTensor =
        graph.meanSquaredCost(this.targetTensor, this.predictionTensor);
    this.session = new Session(graph, this.math);
  }

  public train1Batch(shouldFetchCost: boolean): number {
    this.generateTrainingData();
    // Every 42 steps, lower the learning rate by 15%.
    const learningRate =
        this.initialLearningRate * Math.pow(0.85, Math.floor(this.step++ / 42));
    this.optimizer.setLearningRate(learningRate);
    let costValue = -1;
    this.math.scope(() => {
      const cost = this.session.train(
          this.costTensor, this.feedEntries, this.batchSize, this.optimizer,
          shouldFetchCost ? CostReduction.MEAN : CostReduction.NONE);

      if (!shouldFetchCost) {
        return;
      }
      costValue = cost.get();
    });
    return costValue;
  }

  predict(world: NDArray): Array2D {
    let values = null;
    this.math.scope((keep, track) => {
      const mapping = [{
        tensor: this.inputTensor,
        data: world.reshape([this.size * this.size])
      }]

      const evalOutput = this.session.eval(this.predictionTensor, mapping);
      values = evalOutput.getValues();
    });
    return Array2D.new([this.size, this.size], values);
  }

  private generateTrainingData(): void {
    const batchSize = this.batchSize;
    const size = this.size;

    this.math.scope(() => {
      const inputs = [];
      const outputs = [];
      for (let i = 0; i < batchSize; i++) {
        const example = this.generateGolExample(size);
        inputs.push(example[0].reshape([this.size * this.size]));
        outputs.push(example[1].reshape([this.size * this.size]));
      }

      // TODO(kreeger): Don't really need to shuffle these.
      const inputProviderBuilder =
          new InCPUMemoryShuffledInputProviderBuilder([inputs, outputs]);
      const [inputProvider, targetProvider] =
          inputProviderBuilder.getInputProviders();

      this.feedEntries = [
        {tensor: this.inputTensor, data: inputProvider},
        {tensor: this.targetTensor, data: targetProvider}
      ];
    });
  }

  public generateGolExample(size: number): [NDArray, NDArray] {
    const world = Array2D.randUniform([size - 2, size - 2], 0, 2, 'int32');
    const worldPadded = GameOfLife.padArray(world);
    const numNeighbors = this.countNeighbors(size, worldPadded).getValues();
    const worldValues = world.getValues();
    const nextWorldValues = [];
    for (let i = 0; i < numNeighbors.length; i++) {
      const value = numNeighbors[i];
      let nextVal = 0;
      if (value == 3) {
        // Cell rebirths
        nextVal = 1;
      } else if (value == 2) {
        // Cell survives
        nextVal = worldValues[i];
      } else {
        // Cell dies
        nextVal = 0;
      }
      nextWorldValues.push(nextVal);
    }
    const worldNext = Array2D.new(world.shape, nextWorldValues, 'int32');
    return [worldPadded, GameOfLife.padArray(worldNext)];
  }

  /** Counts total sum of neighbors for a given world. */
  private countNeighbors(size: number, worldPadded: Array2D): Array2D {
    let neighborCount = this.math.add(
        this.math.slice2D(worldPadded, [0, 0], [size - 2, size - 2]),
        this.math.slice2D(worldPadded, [0, 1], [size - 2, size - 2]));
    neighborCount = this.math.add(
        neighborCount,
        this.math.slice2D(worldPadded, [0, 2], [size - 2, size - 2]));
    neighborCount = this.math.add(
        neighborCount,
        this.math.slice2D(worldPadded, [1, 0], [size - 2, size - 2]));
    neighborCount = this.math.add(
        neighborCount,
        this.math.slice2D(worldPadded, [1, 2], [size - 2, size - 2]));
    neighborCount = this.math.add(
        neighborCount,
        this.math.slice2D(worldPadded, [2, 0], [size - 2, size - 2]));
    neighborCount = this.math.add(
        neighborCount,
        this.math.slice2D(worldPadded, [2, 1], [size - 2, size - 2]));
    neighborCount = this.math.add(
        neighborCount,
        this.math.slice2D(worldPadded, [2, 2], [size - 2, size - 2]));
    return neighborCount as Array2D;
  }

  /* Helper method to pad an array until the op is ready. */
  private static padArray(array: NDArray): Array2D<'int32'> {
    const x1 = array.shape[0];
    const x2 = array.shape[1];
    const pad = 1;

    const oldValues = array.getValues();
    const shape = [x1 + pad * 2, x2 + pad * 2];
    const values = [];

    let z = 0;
    for (let i = 0; i < shape[0]; i++) {
      let rangeStart = -1;
      let rangeEnd = -1;
      if (i > 0 && i < shape[0] - 1) {
        rangeStart = i * shape[1] + 1;
        rangeEnd = i * shape[1] + x2;
      }
      for (let j = 0; j < shape[1]; j++) {
        const v = i * shape[0] + j;
        if (v >= rangeStart && v <= rangeEnd) {
          values[v] = oldValues[z++];
        } else {
          values[v] = 0;
        }
      }
    }
    return Array2D.new(shape as [number, number], values, 'int32');
  }

  /* Helper method for creating a fully connected layer. */
  private static createFullyConnectedLayer(
      graph: Graph, inputLayer: Tensor, layerIndex: number,
      sizeOfThisLayer: number, includeRelu = true, includeBias = true) {
    return graph.layers.dense(
        'fully_connected_' + layerIndex, inputLayer, sizeOfThisLayer,
        includeRelu ? (x) => graph.relu(x) : undefined, includeBias);
  }

  /* Helper method for creating a fully connected layer. */
  private static createFullyConnectedLayerSigmoid(
      graph: Graph, inputLayer: Tensor, layerIndex: number,
      sizeOfThisLayer: number, includeBias = true) {
    return graph.layers.dense(
        'fully_connected_' + layerIndex, inputLayer, sizeOfThisLayer,
        (x) => graph.sigmoid(x), includeBias);
  }
}


const game = new GameOfLife(5);
let worlds = game.generateGolExample(5);
game.setupSession();
for (let i = 0; i < 10000; i++) {
  let fetchCost = i % 300 == 0;
  let cost = game.train1Batch(fetchCost);
  if (fetchCost) {
    console.log(i + ': ' + cost);
  }
}
console.log('Game Before:')
testPrint(worlds[0], 5);
console.log('Game After:')
testPrint(worlds[1], 5);
console.log('Prediction:')
testPrint(game.predict(worlds[0]), 5);
console.log('-----------------------------');
console.log('-----------------------------');

for (let i = 0; i < 5; i++) {
  worlds = game.generateGolExample(5);
  console.log('Game Before:')
  testPrint(worlds[0], 5);
  console.log('Game After:')
  testPrint(worlds[1], 5);
  console.log('Prediction:')
  testPrint(game.predict(worlds[0]), 5);
  console.log('-----------------------------');
}