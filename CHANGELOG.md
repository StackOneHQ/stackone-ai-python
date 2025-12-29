# Changelog

## [2.0.0](https://github.com/StackOneHQ/stackone-ai-python/compare/stackone-ai-v0.3.4...stackone-ai-v2.0.0) (2025-12-29)


### ⚠ BREAKING CHANGES

* Drop support for Python 3.9 and 3.10.
* Error handling now uses httpx exceptions instead of requests exceptions. Code catching RequestException should be updated to catch httpx.HTTPStatusError or httpx.RequestError.
* migrate examples and tests to connector-based tool naming ([#51](https://github.com/StackOneHQ/stackone-ai-python/issues/51))
* The `docs` optional dependency group and related commands (`make docs-serve`, `make docs-build`) are no longer available.
* remove MCP server implementation ([#45](https://github.com/StackOneHQ/stackone-ai-python/issues/45))
* remove deprecated OAS-based getTools, migrate to fetchTools only ([#42](https://github.com/StackOneHQ/stackone-ai-python/issues/42))

### Features

* add test coverage reporting with GitHub Pages badge ([#62](https://github.com/StackOneHQ/stackone-ai-python/issues/62)) ([0ef05cf](https://github.com/StackOneHQ/stackone-ai-python/commit/0ef05cf8746e3ca113e6fd6d33bc62fc91711f77))
* **security:** add gitleaks for secret detection ([#63](https://github.com/StackOneHQ/stackone-ai-python/issues/63)) ([1a31baa](https://github.com/StackOneHQ/stackone-ai-python/commit/1a31baa489882da9fc12684d1a060e48928288a9))


### Bug Fixes

* **ci:** use just commands in CI and release workflows ([#57](https://github.com/StackOneHQ/stackone-ai-python/issues/57)) ([38a9dd6](https://github.com/StackOneHQ/stackone-ai-python/commit/38a9dd6cc0b0deea53d1dfb6472686e151df53e4))
* migrate examples and tests to connector-based tool naming ([#51](https://github.com/StackOneHQ/stackone-ai-python/issues/51)) ([c365dbd](https://github.com/StackOneHQ/stackone-ai-python/commit/c365dbd98e8084eea45857292eee90a1798bce16))
* migrate HTTP client from requests to httpx ([#52](https://github.com/StackOneHQ/stackone-ai-python/issues/52)) ([9d180ef](https://github.com/StackOneHQ/stackone-ai-python/commit/9d180efb42647a573e3055ee47ea361dad2dec07))
* remove MCP server implementation ([#45](https://github.com/StackOneHQ/stackone-ai-python/issues/45)) ([bcb12b4](https://github.com/StackOneHQ/stackone-ai-python/commit/bcb12b4ee50e055c4cb29f3aa9baf81352683415))
* **scripts:** add uv lock refresh to version update script ([#50](https://github.com/StackOneHQ/stackone-ai-python/issues/50)) ([bde6d88](https://github.com/StackOneHQ/stackone-ai-python/commit/bde6d88a5688790ada366ed76563092aba0effe4))


### Documentation

* remove meta tools implementation details from README ([#40](https://github.com/StackOneHQ/stackone-ai-python/issues/40)) ([10510d4](https://github.com/StackOneHQ/stackone-ai-python/commit/10510d4b93fc4e20aa51a706541a649115900e6d))
* remove obsolete migration section from README ([#56](https://github.com/StackOneHQ/stackone-ai-python/issues/56)) ([bdcf90d](https://github.com/StackOneHQ/stackone-ai-python/commit/bdcf90d07cd29236f1372d53872689ef624f8e03))


### Miscellaneous Chores

* bump minimum Python version to 3.11 ([#81](https://github.com/StackOneHQ/stackone-ai-python/issues/81)) ([527e828](https://github.com/StackOneHQ/stackone-ai-python/commit/527e8284a73f47af741454610f71d462d095f79c))
* remove MkDocs documentation generation feature ([#46](https://github.com/StackOneHQ/stackone-ai-python/issues/46)) ([947863e](https://github.com/StackOneHQ/stackone-ai-python/commit/947863e91160a07fcd60d8ee837fb79a35abf0b0))
* trigger release 2.0.0 ([#82](https://github.com/StackOneHQ/stackone-ai-python/issues/82)) ([daa963b](https://github.com/StackOneHQ/stackone-ai-python/commit/daa963bacda5d79ad1bc7773d6432507c2ebfdb1))


### Code Refactoring

* remove deprecated OAS-based getTools, migrate to fetchTools only ([#42](https://github.com/StackOneHQ/stackone-ai-python/issues/42)) ([d50d5fb](https://github.com/StackOneHQ/stackone-ai-python/commit/d50d5fb20402dd625217b2900287ae7d9e4cb98c))

## [0.3.4](https://github.com/StackOneHQ/stackone-ai-python/compare/stackone-ai-v0.3.3...stackone-ai-v0.3.4) (2025-11-12)


### Features

* Add MCP-backed dynamic tool fetching to Python SDK ([#39](https://github.com/StackOneHQ/stackone-ai-python/issues/39)) ([d72ca80](https://github.com/StackOneHQ/stackone-ai-python/commit/d72ca808233600bd32374c7e2028232eb54167de))
* add provider/action filtering and hybrid BM25 + TF-IDF search ([#37](https://github.com/StackOneHQ/stackone-ai-python/issues/37)) ([a1c688b](https://github.com/StackOneHQ/stackone-ai-python/commit/a1c688b4efaef9257ecec9827baa7ef90529b9f7))

## [0.3.3](https://github.com/StackOneHQ/stackone-ai-python/compare/stackone-ai-v0.3.2...stackone-ai-v0.3.3) (2025-10-17)


### Features

* feedback tool ([#36](https://github.com/StackOneHQ/stackone-ai-python/issues/36)) ([9179918](https://github.com/StackOneHQ/stackone-ai-python/commit/9179918104c0ec4cfe0488713ca325f0e8e7c6f1))
* LangGraph integration helpers and example ([#33](https://github.com/StackOneHQ/stackone-ai-python/issues/33)) ([983e2f7](https://github.com/StackOneHQ/stackone-ai-python/commit/983e2f7e6551e3722f235ea534ae61f24644350e))


### Bug Fixes

* remove async method ([#31](https://github.com/StackOneHQ/stackone-ai-python/issues/31)) ([370699e](https://github.com/StackOneHQ/stackone-ai-python/commit/370699e390e4a46d8b4ae664fed8f5de6395eb9d))


### Documentation

* use uv for installing ([#30](https://github.com/StackOneHQ/stackone-ai-python/issues/30)) ([3c5d8fb](https://github.com/StackOneHQ/stackone-ai-python/commit/3c5d8fb54e61f8f730098e97f8bf2dfc78cf3bec))

## [0.3.2](https://github.com/StackOneHQ/stackone-ai-python/compare/stackone-ai-v0.3.1...stackone-ai-v0.3.2) (2025-08-26)


### Features

* support Python 3.9+ with optional MCP server support ([#28](https://github.com/StackOneHQ/stackone-ai-python/issues/28)) ([1a37776](https://github.com/StackOneHQ/stackone-ai-python/commit/1a377768c15223e25dbaf1e0bcd0c0e8bb0df2e8))

## [0.3.1](https://github.com/StackOneHQ/stackone-ai-python/compare/stackone-ai-v0.3.0...stackone-ai-v0.3.1) (2025-08-19)


### Documentation

* add comprehensive LangChain integration section to README ([#20](https://github.com/StackOneHQ/stackone-ai-python/issues/20)) ([cbf7f68](https://github.com/StackOneHQ/stackone-ai-python/commit/cbf7f68e889839f9a501ad8f2cd47c468ffff47e))
* rename meta tools ([#27](https://github.com/StackOneHQ/stackone-ai-python/issues/27)) ([a9ebc03](https://github.com/StackOneHQ/stackone-ai-python/commit/a9ebc032f784863913b28d4ad3850b80bafee5f4))

## [0.3.0](https://github.com/StackOneHQ/stackone-ai-python/compare/stackone-ai-v0.0.4...stackone-ai-v0.3.0) (2025-08-19)


### Features

* add CLAUDE.md for Claude Code guidance ([#15](https://github.com/StackOneHQ/stackone-ai-python/issues/15)) ([ac9fe98](https://github.com/StackOneHQ/stackone-ai-python/commit/ac9fe9857f44c19394654dfcbe23fecc5cf9fbb0))
* bring Python SDK to feature parity with Node SDK ([#17](https://github.com/StackOneHQ/stackone-ai-python/issues/17)) ([8b6de99](https://github.com/StackOneHQ/stackone-ai-python/commit/8b6de99184227cb7f1580964dc3eae14f8f60fc1))
* remove automatic STACKONE_ACCOUNT_ID environment variable loading ([#23](https://github.com/StackOneHQ/stackone-ai-python/issues/23)) ([aa0aaf6](https://github.com/StackOneHQ/stackone-ai-python/commit/aa0aaf6d6bf528f8e29def9b008db23cf94b97c7))
* simplify meta tool function names to match Node SDK ([#19](https://github.com/StackOneHQ/stackone-ai-python/issues/19)) ([4572609](https://github.com/StackOneHQ/stackone-ai-python/commit/4572609a9b85a88fc3067be12f821ec0bc54e769))


### Miscellaneous Chores

* release 0.3.0 ([#24](https://github.com/StackOneHQ/stackone-ai-python/issues/24)) ([beea911](https://github.com/StackOneHQ/stackone-ai-python/commit/beea91165ed2ba3eb5f5ad6ca8656344561b0b43))

## [0.0.4](https://github.com/StackOneHQ/stackone-ai-python/compare/stackone-ai-v0.0.3...stackone-ai-v0.0.4) (2025-03-05)


### Bug Fixes

* ci ([#14](https://github.com/StackOneHQ/stackone-ai-python/issues/14)) ([08aba6e](https://github.com/StackOneHQ/stackone-ai-python/commit/08aba6e96e55b4bedc7272e3adc91a1745d7859a))
* script dependencies ([#12](https://github.com/StackOneHQ/stackone-ai-python/issues/12)) ([960c5d8](https://github.com/StackOneHQ/stackone-ai-python/commit/960c5d86f33fcda8bae72d58166ee5991e08f4d5))

## [0.0.3](https://github.com/StackOneHQ/stackone-ai-python/compare/stackone-ai-v0.0.2...stackone-ai-v0.0.3) (2025-03-05)


### Features

* create operations and file upload.  ([#4](https://github.com/StackOneHQ/stackone-ai-python/issues/4)) ([c8469e3](https://github.com/StackOneHQ/stackone-ai-python/commit/c8469e3e0f7d7d35aee88edd0585a76411dcfba1))
* docs ([#3](https://github.com/StackOneHQ/stackone-ai-python/issues/3)) ([13575ea](https://github.com/StackOneHQ/stackone-ai-python/commit/13575eacede3c96ee3861611cdac6fca5663d7e9))
* docs site ([7a2984c](https://github.com/StackOneHQ/stackone-ai-python/commit/7a2984c33deb748abe3f282a449075631da80aef))
* langchain tools ([#2](https://github.com/StackOneHQ/stackone-ai-python/issues/2)) ([c2dc5aa](https://github.com/StackOneHQ/stackone-ai-python/commit/c2dc5aadda1104117c60703ccca6ceb63f8fd68d))
* licence ([#7](https://github.com/StackOneHQ/stackone-ai-python/issues/7)) ([feebc08](https://github.com/StackOneHQ/stackone-ai-python/commit/feebc08ee61f9e4569cbc44c4bac4d1060c036ef))
* openai compat tools ([64ac1da](https://github.com/StackOneHQ/stackone-ai-python/commit/64ac1da8f1d4fad090a1822d751e003a2cca2e52))
* release please ([2946dfb](https://github.com/StackOneHQ/stackone-ai-python/commit/2946dfbdaf2d27bdcfa49925c6aeaa59ea1a9a5e))


### Bug Fixes

* all extras ([235b5e3](https://github.com/StackOneHQ/stackone-ai-python/commit/235b5e32da6a0495d9ef082403ab2898c42a1976))
* Andres comments ([#9](https://github.com/StackOneHQ/stackone-ai-python/issues/9)) ([c97f0b7](https://github.com/StackOneHQ/stackone-ai-python/commit/c97f0b75959f556b94049fc2b65e51172339b718))
* ci ([6aae7fa](https://github.com/StackOneHQ/stackone-ai-python/commit/6aae7fafedebf48a86f6940c479463e1daf4bf93))
* clean up docs ([9186ec3](https://github.com/StackOneHQ/stackone-ai-python/commit/9186ec36937dd4d8cae1fe7367a686aeb01a0459))
* docs formatting ([#6](https://github.com/StackOneHQ/stackone-ai-python/issues/6)) ([2ac9a87](https://github.com/StackOneHQ/stackone-ai-python/commit/2ac9a8792bc630e60bed560102ad55e12f4cc7c5))
* type stubs in python packaging ([#11](https://github.com/StackOneHQ/stackone-ai-python/issues/11)) ([97c6ffe](https://github.com/StackOneHQ/stackone-ai-python/commit/97c6ffed7c6aaaef2834a503013805a6d31836d0))
