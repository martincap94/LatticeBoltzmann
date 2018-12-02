#pragma once
class Texture {
public:

	unsigned int id;
	unsigned int textureUnit;
	int width;
	int height;
	int numChannels;

	Texture();
	Texture(const char *path, unsigned int textureUnit);
	Texture(const char * path, unsigned int textureUnit, bool clampEdges = false);
	~Texture();

	bool loadTexture(const char *path, bool clampEdges = false);

	//bool loadTexture(const char *path);
	void useTexture();
	void use(unsigned int textureUnit);

	void setWrapOptions(unsigned int wrapS, unsigned int wrapT);

};

