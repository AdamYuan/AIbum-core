# AIbum-core
[![WebAssembly](https://github.com/AdamYuan/AIbum-core/actions/workflows/wasm.yml/badge.svg)](https://github.com/AdamYuan/AIbum-core/actions/workflows/wasm.yml)
## React Example
```javascript
import { useState } from "react";
import { simd, threads } from 'wasm-feature-detect';
import ImageUploading from 'react-images-uploading';

let aibumCore;
{ // simple AIbumCore module loader, should implement lazy loading
	let loader;
	const simd_support = await simd();
	const threads_support = await threads();
	if (threads_support)
		loader = simd_support ? import('./core/aibum_core_wasm-simd-threads') : import('./core/aibum_core_wasm-threads');
	else
		loader = simd_support ? import('./core/aibum_core_wasm-simd') : import('./core/aibum_core_wasm-basic');

	const {default : loadAIbumCore} = await loader;
	aibumCore = await loadAIbumCore();
}
const faceNet = new aibumCore.FaceNet();
const imageNet = new aibumCore.ImageNet();

function App() {
	const [images, setImages] = useState([]);

	const [imageFaces, setImageFaces] = useState(null);
	const [imageTags, setImageTags] = useState(null);
	const onChange = (imageList) => {
		setImages(imageList);

		if (imageList.length !== 1) return;

		const file = imageList[0].file;

		let file_reader = new FileReader();
		file_reader.onload = async function () {
			let data = new Uint8Array(file_reader.result);

			// Copy image data to heap
			let heap = aibumCore._malloc(data.length * data.BYTES_PER_ELEMENT);
			aibumCore.HEAPU8.set(data, heap);

			const ab_image = new aibumCore.Image(heap, data.length);

			if (ab_image.valid()) {
				let tags = await imageNet.getTags(ab_image, 5);
				let faces = await faceNet.getFaces(ab_image);

				ab_image.delete(); // Delete image

				tags = await tags.toArray();
				faces = await faces.toArray();
				for (const face of faces)
					face.feature = await face.feature.toArray();

				setImageTags(tags);
				setImageFaces(faces);
			}
		};
		file_reader.readAsArrayBuffer(file);
	};
	return (
		<div className="App">
			<div> { images.length === 0 ? <></> : <img src={images[0]['data_url']} alt="" height="400" /> } </div>
			<ImageUploading multiple={false} value={images} onChange={onChange} maxNumber={1} dataURLKey="data_url">
				{({imageList, onImageUpload, onImageRemoveAll, onImageUpdate, onImageRemove, isDragging, dragProps}) => (
					<div>
						<button style={isDragging ? { color: 'red' } : undefined}
						        onClick={() => imageList.length === 0 ? onImageUpload() : onImageUpdate(0) }
						        {...dragProps}>
							Upload
						</button>
						<button onClick={onImageRemoveAll}>Remove</button>
					</div>
				)}
			</ImageUploading>
			<div> Top 5 tags : </div>
			<div> <textarea readOnly value={JSON.stringify(imageTags)}/> </div>
			<div> Detected {imageFaces === null ? 0 : imageFaces.length} faces : </div>
			<div> <textarea readOnly value={JSON.stringify(imageFaces)}/> </div>
		</div>
	);
}

export default App;
```
