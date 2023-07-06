# AIbum-core
[![WebAssembly](https://github.com/AdamYuan/AIbum-core/actions/workflows/wasm.yml/badge.svg)](https://github.com/AdamYuan/AIbum-core/actions/workflows/wasm.yml)
## React Example
```javascript
import { useState } from "react";
import ImageUploading from 'react-images-uploading';
import loadAIbumCore from './core/AIbumCore';

const aibumCore = await loadAIbumCore();
const faceNet = new aibumCore.FaceNet();
const imageNet = new aibumCore.ImageNet();
const styleTransfer = new aibumCore.StyleTransfer();

function getStyleModel(uri) {
    return new Promise((resolve) => {
        fetch(uri).then(
            async (res) => {
                var fr = new FileReader();
                fr.onload = (e) => {
                    resolve(new Uint8Array(e.target.result));
                };
                fr.onerror = () => { resolve(null); };
                fr.readAsArrayBuffer(await res.blob());
            },
            () => { resolve(null); }
        );
    });
}

function App() {
    const [images, setImages] = useState([]);
    const [abImage, setABImage] = useState(null);

    const [imageFaces, setImageFaces] = useState(null);
    const [imageTags, setImageTags] = useState(null);

    const onChange = (imageList) => {
        if (imageList.length !== 1) return;

        const file = imageList[0].file;

        let file_reader = new FileReader();
        file_reader.onload = async (e) => {
            const ab_image = new aibumCore.Image(new Uint8Array(e.target.result));
            if (ab_image.valid) {
                setImages(imageList);
                setABImage(prev => {
                    if (prev !== null)
                        prev.free();
                    return ab_image;
                });

                const tags = await imageNet.getTags(ab_image, 5);
                const faces = await faceNet.getFaces(ab_image);

                setImageTags(tags);
                setImageFaces(faces);
            }
        };
        file_reader.readAsArrayBuffer(file);
    };

    const onStyleChange = async (e) => {
        if (e.target.value !== "") {
            const model = await getStyleModel("./styles/" + e.target.value + ".bin");
            if (model !== null)
                styleTransfer.load(model);
        } else
            styleTransfer.free();
    }

    const onStyleTransfer = async () => {
        if (abImage === null || !abImage.valid)
            return;
        const styled = await styleTransfer.transfer(abImage, 512);
        if (!styled.valid)
            return;

        const canvas = document.getElementById("transfered");
        const ctx = canvas.getContext("2d");
        ctx.canvas.width = styled.width;
        ctx.canvas.height = styled.height;
        let imageData = ctx.createImageData(styled.width, styled.height);
        imageData.data.set(styled.data);
        ctx.putImageData(imageData, 0, 0);

        styled.free();
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
                    </div>
                )}
            </ImageUploading>
            <div> Top 5 tags : </div>
            <div> <textarea readOnly value={JSON.stringify(imageTags)}/> </div>
            <div> Detected {imageFaces === null ? 0 : imageFaces.length} faces : </div>
            <div> <textarea readOnly value={JSON.stringify(imageFaces)}/> </div>
            <div>
                <select name="styles" id="style-select" onChange={onStyleChange}>
                    <option value="">--</option>
                    <option value="candy">Candy</option>
                    <option value="mosaic">Mosaic</option>
                    <option value="pointilism">Pointilism</option>
                    <option value="rain_princess">Rain Princess</option>
                    <option value="udnie">Udnie</option>
                </select>
                <button onClick={onStyleTransfer}> style transfer </button>
            </div>
            <canvas
                id="transfered"
            >
                Your browser does not support the HTML canvas tag.
            </canvas>
        </div>
    );
}

export default App;
```
