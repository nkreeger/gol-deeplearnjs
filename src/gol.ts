import {Array2D} from 'deeplearn';
import { NDArray } from 'deeplearn/dist/math/ndarray';

/**
 * 
 * @param array 
 * @param padding 
 */
function padArray(array: NDArray): Array2D<'int32'> {
  const x1 = array.shape[0];
  const x2 = array.shape[1];
  const pad = 1;

  const oldValues = array.getValues();
  const shape = [x1 + pad * 2, x2 + pad * 2];
  const values = [];

  let z = 0;
  for (let i = 0; i < shape[0]; i++) {
    let rangeStart = i * shape[1] + 1;
    let rangeEnd = i * shape[1] + x2;
    for (let j = 0; j < shape[1]; j++) {
      const v = i * shape[0] + j;
      if (i > 0 && i < shape[0] -1 && v >= rangeStart && v <= rangeEnd) {
        values[v] = oldValues[z++];
      } else {
        values[v] = 0;
      }
    }
  }
  return Array2D.new(shape as [number, number], values, 'int32');
}

/**
 * 
 * @param size 
 */
function generateGolExample(size: number) {
  if (size < 3) {
    new Error('Size must be greater than 2');
  }

  const world = Array2D.randUniform([size - 2, size -2], 0, 2, 'int32');
  const worldPadded = padArray(world);
}

/**
 * 
 * @param worldNextProbs 
 * @param worldNextTarget 
 */
function golLoss(worldNextProbs: NDArray, worldNextTarget: NDArray) {
  // TODO: Calculate loss.
}

/**
 * 
 * @param size 
 */
function trainModel(size: number) {
  const world = generateGolExample(size);
  const worldNext = generateGolExample(size);

  // TODO: input-layer
  // TODO: target
  // TODO: fully-connect layers
}

// Start:
trainModel(3);
trainModel(5);

// TODO: Run the graph.