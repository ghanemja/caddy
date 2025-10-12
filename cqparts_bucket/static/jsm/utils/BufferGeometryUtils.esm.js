// ESM adapter for the UMD utils you have
import * as THREE from '../three.module.js';

// paste ONLY the functions you need from your UMD file, converted to ESM:
function toTrianglesDrawMode( geometry, drawMode ) {
  if ( drawMode === THREE.TrianglesDrawMode ) {
    console.warn('Geometry already triangles.');
    return geometry;
  }
  if ( drawMode === THREE.TriangleFanDrawMode || drawMode === THREE.TriangleStripDrawMode ) {
    let index = geometry.getIndex();
    if ( index === null ) {
      const indices = [];
      const position = geometry.getAttribute('position');
      if (!position) { console.error('No position attribute.'); return geometry; }
      for ( let i = 0; i < position.count; i++ ) indices.push(i);
      geometry.setIndex(indices);
      index = geometry.getIndex();
    }
    const numberOfTriangles = index.count - 2;
    const newIndices = [];
    if ( drawMode === THREE.TriangleFanDrawMode ) {
      for ( let i = 1; i <= numberOfTriangles; i++ ) {
        newIndices.push( index.getX(0), index.getX(i), index.getX(i+1) );
      }
    } else {
      for ( let i = 0; i < numberOfTriangles; i++ ) {
        if ( (i % 2) === 0 ) newIndices.push( index.getX(i), index.getX(i+1), index.getX(i+2) );
        else                 newIndices.push( index.getX(i+2), index.getX(i+1), index.getX(i) );
      }
    }
    const newGeometry = geometry.clone();
    newGeometry.setIndex(newIndices);
    newGeometry.clearGroups();
    return newGeometry;
  }
  console.error('Unknown draw mode:', drawMode);
  return geometry;
}

export { toTrianglesDrawMode };
