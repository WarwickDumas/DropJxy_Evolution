

if (1) {

	GlobalCutaway = false;

	// 1.  Represent epsilon xy as xy velocity

	cudaMemcpy(p_tempvec2_host, p_epsilon_xy, sizeof(f64_vec2)*NMINOR, cudaMemcpyDeviceToHost);
	f64 maxx = 0.0, minx = 0.0, miny = 0.0, maxy = 0.0;
	long iMin = 0, iMax = 0, iMiy = 0, iMay = 0;
	for (iMinor = 0; iMinor < NMINOR; iMinor++)
	{
		if (p_tempvec2_host[iMinor].x > maxx) {
			maxx = p_tempvec2_host[iMinor].x;
			iMax = iMinor;
		};
		if (p_tempvec2_host[iMinor].x < minx) {
			minx = p_tempvec2_host[iMinor].x;
			iMin = iMinor;
		};
		if (p_tempvec2_host[iMinor].y > maxy) {
			maxy = p_tempvec2_host[iMinor].y;
			iMay = iMinor;
		};
		if (p_tempvec2_host[iMinor].y < miny) {
			miny = p_tempvec2_host[iMinor].y;
			iMiy = iMinor;
		};
	};
	printf("Results of prev move: epsilon\n");
	printf("maxx %1.9E at %d ; minx %1.9E at %d\n", maxx, iMax, minx, iMin);
	printf("maxy %1.9E at %d ; miny %1.9E at %d\n", maxy, iMay, miny, iMiy);

	// 2.  epsilon ez graph
	cudaMemcpy(p_temphost1, p_epsilon_ez, sizeof(f64)*NMINOR, cudaMemcpyDeviceToHost);
	f64 maxz = 0.0, minz = 0.0;
	long iMiz = 0, iMaz = 0;
	for (iMinor = 0; iMinor < NMINOR; iMinor++)
	{
		if (p_temphost1[iMinor] > maxz) {
			maxz = p_temphost1[iMinor];
			iMaz = iMinor;
		};
		if (p_temphost1[iMinor] < minz) {
			minz = p_temphost1[iMinor];
			iMiz = iMinor;
		};
	};
	printf("maxz %1.9E at %d ; minz %1.9E at %d\n", maxz, iMaz, minz, iMiz);

	long iVertex;
	char buffer[256];
	Vertex * pVertex = pTriMesh->X;
	plasma_data * pdata = pTriMesh->pData + BEGINNING_OF_CENTRAL;
	for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
	{
		pdata->temp.x = p_tempvec2_host[iVertex + BEGINNING_OF_CENTRAL].x;
		pdata->temp.y = p_tempvec2_host[iVertex + BEGINNING_OF_CENTRAL].y;
		++pVertex;
		++pdata;
	}
	sprintf(buffer, "epsilon_xy %d", iIteration);
	Graph[0].DrawSurface(buffer,
		VELOCITY_HEIGHT, (f64 *)(&((pTriMesh->pData[0]).temp.x)),
		VELOCITY_COLOUR, (f64 *)(&((pTriMesh->pData[0]).temp.x)),
		false,
		GRAPH_EPSILON, pTriMesh);

	pVertex = pTriMesh->X;
	pdata = pTriMesh->pData + BEGINNING_OF_CENTRAL;
	for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
	{
		pdata->temp.x = p_temphost1[iVertex + BEGINNING_OF_CENTRAL];
		pdata->temp.y = p_temphost1[iVertex + BEGINNING_OF_CENTRAL];
		++pVertex;
		++pdata;
	}
	sprintf(buffer, "epsilon_z %d", iIteration);
	Graph[1].DrawSurface(buffer,
		DATA_HEIGHT, (f64 *)(&((pTriMesh->pData[0]).temp.x)),
		AZSEGUE_COLOUR, (f64 *)(&((pTriMesh->pData[0]).temp.x)),
		false,
		GRAPH_EZ, pTriMesh);

	// Graph 2 : vie_k+1 xy - vie_k xy

	cudaMemcpy(p_v4host, p_vie, sizeof(v4)*NMINOR, cudaMemcpyDeviceToHost);
	cudaMemcpy(p_v4host2, p_vie_k, sizeof(v4)*NMINOR, cudaMemcpyDeviceToHost);

	printf("Graphing vxy diff...\n");

	Vertex * pVertex = pTriMesh->X;
	plasma_data * pdata = pTriMesh->pData + BEGINNING_OF_CENTRAL;
	for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
	{
		pdata->temp.x = p_v4host[iVertex + BEGINNING_OF_CENTRAL].vxy.x - p_v4host2[iVertex + BEGINNING_OF_CENTRAL].vxy.x;
		pdata->temp.y = p_v4host[iVertex + BEGINNING_OF_CENTRAL].vxy.y - p_v4host2[iVertex + BEGINNING_OF_CENTRAL].vxy.y;
		++pVertex; ++pdata;
	};
	sprintf(buffer, "vmove_xy %d", iIteration);
	Graph[2].DrawSurface(buffer,
		VELOCITY_HEIGHT, (f64 *)(&((pTriMesh->pData[0]).temp.x)),
		VELOCITY_COLOUR, (f64 *)(&((pTriMesh->pData[0]).temp.x)),
		false,
		GRAPH_ELEC_V, pTriMesh);

	// Graph 3 : vie_vez - vie_k vez

	Vertex * pVertex = pTriMesh->X;
	plasma_data * pdata = pTriMesh->pData + BEGINNING_OF_CENTRAL;
	for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
	{
		pdata->temp.x = p_v4host[iVertex + BEGINNING_OF_CENTRAL].vez - p_v4host2[iVertex + BEGINNING_OF_CENTRAL].vez;
		++pVertex; ++pdata;
	};
	sprintf(buffer, "vmove_vez %d", iIteration);
	Graph[3].DrawSurface(buffer,
		DATA_HEIGHT, (f64 *)(&((pTriMesh->pData[0]).temp.x)),
		AZSEGUE_COLOUR, (f64 *)(&((pTriMesh->pData[0]).temp.x)),
		false,
		GRAPH_VEZ, pTriMesh);

	printf("Graphing vez diff...\n");

	// Graph 4 : heating : T change





	// Graph 5 : ?


