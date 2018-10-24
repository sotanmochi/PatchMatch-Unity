using UnityEngine;
using UnityEngine.Rendering;

public class PatchMatchGPU : MonoBehaviour
{
    public ComputeShader computeShader;
	public Material inputImageMaterialA;
	public Material inputImageMaterialB;
	public Material outputFlowmapMaterial;
	public Material outputReconstructedImageMaterial;
    public TextMesh fpsText;

	// --------------------------------------------
	//          Parameters for PatchMatch
	// --------------------------------------------
	private int patch_w = 3; // suppose patch_w is an odd number
	private int pm_iters = 5;
	private int rs_max = int.MaxValue;
	// --------------------------------------------

	private Texture2D srcImageA;
	private Texture2D srcImageB;
	private RenderTexture flowmap;
	private RenderTexture reconstructed;

	private ComputeBuffer annBuffer;
	private ComputeBuffer anndBuffer;
	private ComputeBuffer paramsBuffer;

	private uint[] _ann;
	private float[] _annd;
	private int[] _params;

	void Start()
	{
		srcImageA = (Texture2D)inputImageMaterialA.mainTexture;
		srcImageB = (Texture2D)inputImageMaterialB.mainTexture;

        flowmap = new RenderTexture(srcImageA.width, srcImageA.height, 0, RenderTextureFormat.ARGB32);
        flowmap.enableRandomWrite = true;
        flowmap.Create();

        reconstructed = new RenderTexture(srcImageB.width, srcImageB.height, 0, RenderTextureFormat.ARGB32);
        reconstructed.enableRandomWrite = true;
        reconstructed.Create();

		int a_rows = srcImageA.height;
		int a_cols = srcImageA.width;
		int b_rows = srcImageB.height;
		int b_cols = srcImageB.width;

		int sizeOfParams = 7;
		_params = new int[sizeOfParams];
		_params[0] = a_rows;
		_params[1] = a_cols;
		_params[2] = b_rows;
		_params[3] = b_cols;
		_params[4] = patch_w;
		_params[5] = pm_iters;
		_params[6] = rs_max;
		
		int sizeOfAnn = a_rows*a_cols;
		_ann = new uint[sizeOfAnn];
		_annd = new float[sizeOfAnn];

		annBuffer = new ComputeBuffer(sizeOfAnn, sizeof(uint));
		anndBuffer = new ComputeBuffer(sizeOfAnn, sizeof(float));
		paramsBuffer = new ComputeBuffer(sizeOfParams, sizeof(int));
	}

	void Update()
	{
        if (!SystemInfo.supportsComputeShaders)
        {
			Debug.LogError("Compute Shader is not Support!!");
			return;
        }
		if (computeShader == null)
		{
			Debug.LogError("Compute Shader has not been assigned!!");
			return;
		}
        if (fpsText != null && Time.frameCount % 10 == 0)
        {
            fpsText.text = "FPS: " + Mathf.Round(1.0f/Time.unscaledDeltaTime).ToString();;
        }

		int kernelID = -1;
		
        annBuffer.SetData(_ann);
		anndBuffer.SetData(_annd);
		paramsBuffer.SetData(_params);

		kernelID = computeShader.FindKernel("PatchMatchCS");
		computeShader.SetBuffer(kernelID, "_ann", annBuffer);
		computeShader.SetBuffer(kernelID, "_annd", anndBuffer);
		computeShader.SetBuffer(kernelID, "_params", paramsBuffer);
		computeShader.SetTexture(kernelID, "_srcImageA", srcImageA);
		computeShader.SetTexture(kernelID, "_srcImageB", srcImageB);
        computeShader.SetTexture(kernelID, "_flowmap", flowmap);
        computeShader.SetTexture(kernelID, "_reconstructedImage", reconstructed);
        computeShader.Dispatch(kernelID, srcImageA.width/8 + 1, srcImageA.height/8 + 1, 1);

		outputFlowmapMaterial.mainTexture = flowmap;
		outputReconstructedImageMaterial.mainTexture = reconstructed;
	}

	void OnDestroy()
	{
		annBuffer.Release();
		anndBuffer.Release();
		paramsBuffer.Release();
	}
}
