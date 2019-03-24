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
	private int gpu_prop = 8;

	private int a_height;
	private int a_width;
	private int b_height;
	private int b_width;
	// --------------------------------------------

	private Texture2D srcImageA;
	private Texture2D srcImageB;
	private RenderTexture flowmap;
	private RenderTexture reconstructed;

	private RenderTexture annBuffer;
	private RenderTexture anndBuffer;
	private RenderTexture annOutBuffer;
	private RenderTexture anndOutBuffer;

	void Start()
	{
		srcImageA = (Texture2D)inputImageMaterialA.mainTexture;
		srcImageB = (Texture2D)inputImageMaterialB.mainTexture;

		annBuffer = new RenderTexture(srcImageA.width, srcImageA.height, 0, RenderTextureFormat.RGInt);
        annBuffer.enableRandomWrite = true;
        annBuffer.Create();

		anndBuffer = new RenderTexture(srcImageA.width, srcImageA.height, 0, RenderTextureFormat.RFloat);
        anndBuffer.enableRandomWrite = true;
        anndBuffer.Create();

		annOutBuffer = new RenderTexture(srcImageA.width, srcImageA.height, 0, RenderTextureFormat.RGInt);
        annOutBuffer.enableRandomWrite = true;
        annOutBuffer.Create();

		anndOutBuffer = new RenderTexture(srcImageA.width, srcImageA.height, 0, RenderTextureFormat.RFloat);
        anndOutBuffer.enableRandomWrite = true;
        anndOutBuffer.Create();

        flowmap = new RenderTexture(srcImageA.width, srcImageA.height, 0, RenderTextureFormat.ARGB32);
        flowmap.enableRandomWrite = true;
        flowmap.Create();

        reconstructed = new RenderTexture(srcImageA.width, srcImageA.height, 0, RenderTextureFormat.ARGB32);
        reconstructed.enableRandomWrite = true;
        reconstructed.Create();

		a_height = srcImageA.height;
		a_width = srcImageA.width;
		b_height = srcImageB.height;
		b_width = srcImageB.width;
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

		InvokePatchMatch();
	}

	void InvokePatchMatch()
	{
		int kernelID = -1;

        // *************************
        //   Initialize parameters
        // *************************
        computeShader.SetInt("_a_height", a_height);
        computeShader.SetInt("_a_width", a_width);
        computeShader.SetInt("_b_height", b_height);
        computeShader.SetInt("_b_width", b_width);
        computeShader.SetInt("_patch_w", patch_w);
        computeShader.SetInt("_pm_iters", pm_iters);
        computeShader.SetInt("_rs_max", rs_max);

        // ************************************************
        //   Initialize Approximate Nearest Neighbor(ANN)
        // ************************************************
		kernelID = computeShader.FindKernel("InitializeAnnCS");
		computeShader.SetTexture(kernelID, "_SrcImageA", srcImageA);
		computeShader.SetTexture(kernelID, "_SrcImageB", srcImageB);
		computeShader.SetTexture(kernelID, "_Ann", annBuffer);
		computeShader.SetTexture(kernelID, "_Annd", anndBuffer);
        computeShader.SetTexture(kernelID, "_Flowmap", flowmap);
        computeShader.Dispatch(kernelID, srcImageA.width/8 + 1, srcImageA.height/8 + 1, 1);

		// ************************
        //   PatchMatch iteration
        // ************************
		for (int iter = 0; iter < pm_iters; iter++)
		{
			/* Propagation */
			for (int jump = gpu_prop; jump >= 1; jump /= 2)
			{
				kernelID = computeShader.FindKernel("PropagationCS");
        		computeShader.SetInt("_prop_jump", jump);
				computeShader.SetTexture(kernelID, "_SrcImageA", srcImageA);
				computeShader.SetTexture(kernelID, "_SrcImageB", srcImageB);
				computeShader.SetTexture(kernelID, "_Ann", annBuffer);
				computeShader.SetTexture(kernelID, "_Annd", anndBuffer);
				computeShader.SetTexture(kernelID, "_AnnOut", annOutBuffer);
				computeShader.SetTexture(kernelID, "_AnndOut", anndOutBuffer);
				computeShader.Dispatch(kernelID, srcImageA.width/8 + 1, srcImageA.height/8 + 1, 1);

				SwapBuffer(ref annBuffer, ref annOutBuffer);
				SwapBuffer(ref anndBuffer, ref anndOutBuffer);
			}

			/* Random search */
			kernelID = computeShader.FindKernel("RandomSearchCS");
			computeShader.SetTexture(kernelID, "_SrcImageA", srcImageA);
			computeShader.SetTexture(kernelID, "_SrcImageB", srcImageB);
			computeShader.SetTexture(kernelID, "_Ann", annBuffer);
			computeShader.SetTexture(kernelID, "_Annd", anndBuffer);
	        computeShader.Dispatch(kernelID, srcImageA.width/8 + 1, srcImageA.height/8 + 1, 1);
		}

        // *********************
        //   Reconstruct Image
        // *********************
		kernelID = computeShader.FindKernel("ReconstructImageCS");
		computeShader.SetTexture(kernelID, "_SrcImageA", srcImageA);
		computeShader.SetTexture(kernelID, "_SrcImageB", srcImageB);
		computeShader.SetTexture(kernelID, "_Ann", annBuffer);
		computeShader.SetTexture(kernelID, "_Flowmap", flowmap);
		computeShader.SetTexture(kernelID, "_ReconstructedImage", reconstructed);
        computeShader.Dispatch(kernelID, srcImageA.width/8 + 1, srcImageA.height/8 + 1, 1);

        // *******************
        //       Output
        // *******************
		outputFlowmapMaterial.mainTexture = flowmap;
		outputReconstructedImageMaterial.mainTexture = reconstructed;
	}

	void SwapBuffer(ref RenderTexture ping, ref RenderTexture pong) {
		RenderTexture temp = ping;
		ping = pong;
		pong = temp;
	}
}
