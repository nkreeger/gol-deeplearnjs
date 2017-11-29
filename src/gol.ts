import {Array2D} from 'deeplearn';

function generateGolExample(size: number) {
  if (size < 3) {
    new Error('Size must be greater than 2');
  }

  const world = Array2D.randUniform([size - 2, size -2], 0, 2, 'int32');
  console.log('world', world.getValues());
}

// Start:
generateGolExample(4);
