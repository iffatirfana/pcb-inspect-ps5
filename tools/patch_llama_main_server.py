import pathlib, sys
p = pathlib.Path(sys.argv[1]); s = p.read_text()
assert "--server" not in s and "FLAGS_server" not in s, "already patched"

old = "#include <fstream>\n"
assert old in s
s = s.replace(old, old + "#include <iostream>\n", 1)

old = "// Helper function to parse comma-separated string lists\n"
assert old in s
s = s.replace(old, '''DEFINE_bool(
    server,
    false,
    "Keep the model loaded. Read one prompt-file path per line from stdin, answer each, print <<LLM_DONE>>.");

''' + old, 1)

old = "  if (warmup) {\n"
assert s.count(old) == 1
s = s.replace(old, '''  if (FLAGS_server) {
    if (runner->load() != executorch::runtime::Error::Ok) {
      ET_LOG(Error, "Failed to load llama runner");
      return 1;
    }
    printf("<<LLM_READY>>\\n");
    fflush(stdout);
    std::string req_path;
    while (std::getline(std::cin, req_path)) {
      if (req_path.empty() || req_path == "quit") {
        break;
      }
      std::string req_prompt;
      if (!readFileToString(req_path, req_prompt)) {
        printf("\\n<<LLM_ERROR>>\\n");
        fflush(stdout);
        continue;
      }
      runner->reset();
      executorch::extension::llm::GenerationConfig req_config{};
      req_config.temperature = temperature;
      req_config.ignore_eos = FLAGS_ignore_eos;
      req_config.num_bos = FLAGS_num_bos;
      req_config.num_eos = FLAGS_num_eos;
      if (FLAGS_max_new_tokens != -1) {
        req_config.max_new_tokens = FLAGS_max_new_tokens;
      } else {
        req_config.seq_len = seq_len;
      }
      auto req_error = runner->generate(req_prompt, req_config);
      fflush(stdout);
      printf(
          req_error == executorch::runtime::Error::Ok ? "\\n<<LLM_DONE>>\\n"
                                                      : "\\n<<LLM_ERROR>>\\n");
      fflush(stdout);
    }
    return 0;
  }

''' + old, 1)
p.write_text(s)
print("patched", p)
