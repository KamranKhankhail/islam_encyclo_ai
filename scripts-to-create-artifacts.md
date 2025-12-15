analyse & review these step by step, steps to generate embeddings and other artifacts required for our lexical + semantics search in offline mobile local feature, and generate a comprehensive documentation for junior engineers to follow if they want to generate artifacts from scratch:

1. python scripts/prepare_data.py
2. pytest tests/test_data.py -v
3. python .\scripts\generate_embeddings_e5.py
4. python tools/build_askquran_pack_v1.py
5. Add Pack v1 to the RN app bundle
    android/app/src/main/assets/askquran/v1/
    ios/AskQuranAssets/askquran/v1/
6. Add a single RN debug function to verify the pack exists
7. 