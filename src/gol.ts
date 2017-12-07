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

import {Array2D, Graph, NDArray, NDArrayMathGPU, Session} from 'deeplearn';
import {Tensor} from 'deeplearn/dist/graph/graph';
import {Scalar} from 'deeplearn/dist/math/ndarray';
import { NDArrayMath } from 'deeplearn/dist/math/math';
import { FeedEntry } from 'deeplearn/dist/graph/session';
import { Server } from 'http';

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
 * Main class for running a deep-neural network of training for Game-of-life next sequence.
 */
class GameOfLife {
  session: Session;
  math: NDArrayMath = new NDArrayMathGPU();
  batchSize: 300;

  inputTensor: Tensor;
  targetTensor: Tensor;
  costTensor: Tensor;
  predictionTensor: Tensor;

  size: number;

  // Maps tensors to InputProviders
  feedEntries: FeedEntry[];

  constructor(size: number) {
    this.size = size;
  }

  setupSession(): void {
    const graph = new Graph();
    const size = this.size * this.size;

    this.inputTensor = graph.placeholder('input', [size]);
    this.targetTensor = graph.placeholder('target', [size]);

    let hiddenLayer = GameOfLife.createFullyConnectedLayer(graph, this.inputTensor, 0, size);
    hiddenLayer = GameOfLife.createFullyConnectedLayer(graph, hiddenLayer, 1, size);
    // This needs activiation function sigmoid?
    hiddenLayer = GameOfLife.createFullyConnectedLayer(graph, hiddenLayer, 2, size);

    // TODO(kreeger): Left off right here.
    // tf.contrib.losses.log_loss()
    // this.predictionTensor = GameOfLife.createFullyConnectedLayer(graph, hiddenLayer, 3, )
    this.costTensor = graph.meanSquaredCost(this.targetTensor, this.predictionTensor);


    this.session = new Session(graph, this.math);

    // Generate the training data:
    this.generateTrainingData();
  }

  private generateTrainingData(): void {
    // TODO(kreeger): Simply pipe the information below into the correct Feed.
  }

  private generateGolExample(size: number) {
    // TODO(kreeger): Optimize this until vectorization is available.
    const world = Array2D.randUniform([size - 2, size - 2], 0, 2, 'int32');
    const worldPadded = GameOfLife.padArray(world);
    const numNeighbors = this.countNeighbors(size, worldPadded);
    const survivors = GameOfLife.cellSurvivors(world, numNeighbors);
    const rebirths = GameOfLife.cellRebirths(world, numNeighbors);
    const worldNext = GameOfLife.createNextWorld(world, survivors, rebirths);
    return [worldPadded, GameOfLife.padArray(worldNext)]; 
  }

  /** Counts total sum of neighbors for a given world. */
  private countNeighbors(size: number, worldPadded: Array2D): Array2D {
    let neighborCount = this.math.add(
        this.math.slice2D(worldPadded, [0, 0], [size - 2, size - 2]),
        this.math.slice2D(worldPadded, [0, 1], [size - 2, size - 2]));
    neighborCount = this.math.add(
        neighborCount, this.math.slice2D(worldPadded, [0, 2], [size - 2, size - 2]));
    neighborCount = this.math.add(
        neighborCount, this.math.slice2D(worldPadded, [1, 0], [size - 2, size - 2]));
    neighborCount = this.math.add(
        neighborCount, this.math.slice2D(worldPadded, [1, 2], [size - 2, size - 2]));
    neighborCount = this.math.add(
        neighborCount, this.math.slice2D(worldPadded, [2, 0], [size - 2, size - 2]));
    neighborCount = this.math.add(
        neighborCount, this.math.slice2D(worldPadded, [2, 1], [size - 2, size - 2]));
    neighborCount = this.math.add(
        neighborCount, this.math.slice2D(worldPadded, [2, 2], [size - 2, size - 2]));
    return neighborCount as Array2D;
  }

  /** Generates cell survivors matrix. */
  private static cellSurvivors(world: Array2D, numNeigbors: Array2D): Array2D {
    const survives = [];
    let worldValues = world.getValues();
    let numNeigborValues = numNeigbors.getValues();
    for (let i = 0; i < numNeigborValues.length; i++) {
      const value = numNeigborValues[i];
      value == 2 || value == 3 ? survives.push(worldValues[i]) : survives.push(0);
    }
    return Array2D.new(world.shape, survives, 'bool');
  }

  /** Generates an array if the cell should rebirth. */
  private static cellRebirths(world: Array2D, numNeigbors: Array2D): Array2D {
    const rebirths = []
    let numNeigborValues = numNeigbors.getValues();
    for (let i = 0; i < numNeigborValues.length; i++) {
      numNeigborValues[i] == 3 ? rebirths.push(1) : rebirths.push(0);
    }
    return Array2D.new(world.shape, rebirths, 'bool');
  }

  /** Generates the next world sequence. */
  private static createNextWorld(world: Array2D, survives: Array2D, rebirths: Array2D): Array2D {
    const surviveValues = survives.getValues();
    const rebirthValues = rebirths.getValues();
    const nextWorldValues = [];
    for (let i = 0; i < surviveValues.length; i++) {
      if (rebirthValues[i]) {
        nextWorldValues.push(1);
      } else {
        nextWorldValues.push(surviveValues[i]);
      }
    }
    return Array2D.new(world.shape, nextWorldValues, 'int32');
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
}

const game = new GameOfLife(5);
game.setupSession();

// TODO - start?