/**
 * Prompt Handler - Manages MCP prompt requests
 * Handles prompt listing and retrieval through the service registry
 */

export class PromptHandler {
  constructor(serviceRegistry, logger) {
    this.serviceRegistry = serviceRegistry;
    this.logger = logger;
    this.promptServices = new Map();
    this.isInitialized = false;
  }

  /**
   * Initialize the prompt handler
   */
  async initialize() {
    try {
      this.logger.info('Initializing Prompt Handler');

      // Discover and register prompt services
      await this.discoverPromptServices();

      this.isInitialized = true;
      this.logger.info('Prompt Handler initialized successfully');

    } catch (error) {
      this.logger.error('Failed to initialize Prompt Handler:', error);
      throw error;
    }
  }

  /**
   * Discover prompt services from the service registry
   */
  async discoverPromptServices() {
    // Find services that provide prompts
    const promptProviders = this.serviceRegistry.findByCapability('prompts');

    this.logger.info(`Discovered ${promptProviders.length} prompt provider services`);

    for (const { name, service, metadata } of promptProviders) {
      try {
        // Get prompts from the service
        const prompts = await this.getPromptsFromService(service);

        if (prompts && prompts.length > 0) {
          this.promptServices.set(name, {
            service,
            prompts,
            metadata
          });

          this.logger.info(`Registered ${prompts.length} prompts from service: ${name}`);
        }

      } catch (error) {
        this.logger.error(`Failed to get prompts from service ${name}:`, error);
      }
    }
  }

  /**
   * Get prompts from a service
   */
  async getPromptsFromService(service) {
    if (typeof service.getPrompts === 'function') {
      return await service.getPrompts();
    }

    if (typeof service.listPrompts === 'function') {
      return await service.listPrompts();
    }

    if (service.prompts && Array.isArray(service.prompts)) {
      return service.prompts;
    }

    return [];
  }

  /**
   * List all available prompts
   */
  async listPrompts(request) {
    if (!this.isInitialized) {
      throw new Error('Prompt handler not initialized');
    }

    try {
      const allPrompts = [];

      // Collect prompts from all registered services
      for (const [serviceName, { prompts, metadata }] of this.promptServices) {
        for (const prompt of prompts) {
          allPrompts.push({
            ...prompt,
            _service: serviceName,
            _serviceType: metadata.type,
            _serviceVersion: metadata.version
          });
        }
      }

      this.logger.info(`Listed ${allPrompts.length} prompts from ${this.promptServices.size} services`);

      return {
        prompts: allPrompts
      };

    } catch (error) {
      this.logger.error('Failed to list prompts:', error);
      throw error;
    }
  }

  /**
   * Get a specific prompt
   */
  async getPrompt(request) {
    if (!this.isInitialized) {
      throw new Error('Prompt handler not initialized');
    }

    try {
      const { name: promptName, arguments: promptArgs } = request.params;

      if (!promptName) {
        throw new Error('Prompt name is required');
      }

      this.logger.info(`Getting prompt: ${promptName}`, { args: promptArgs });

      // Find the service that provides this prompt
      const { service, serviceName } = await this.findPromptService(promptName);

      if (!service) {
        throw new Error(`Prompt '${promptName}' not found`);
      }

      // Get the prompt
      const result = await this.getPromptFromService(service, promptName, promptArgs);

      this.logger.info(`Prompt '${promptName}' retrieved successfully`);

      return result;

    } catch (error) {
      this.logger.error(`Failed to get prompt '${request.params.name}':`, error);
      throw error;
    }
  }

  /**
   * Find the service that provides a specific prompt
   */
  async findPromptService(promptName) {
    for (const [serviceName, { service, prompts }] of this.promptServices) {
      const prompt = prompts.find(p => p.name === promptName);
      if (prompt) {
        return { service, serviceName, prompt };
      }
    }

    return { service: null, serviceName: null, prompt: null };
  }

  /**
   * Get a prompt from a specific service
   */
  async getPromptFromService(service, promptName, promptArgs) {
    // Try different method names for prompt retrieval
    const getMethods = [
      'getPrompt',
      'retrievePrompt',
      'fetchPrompt',
      'loadPrompt',
      'generatePrompt'
    ];

    for (const methodName of getMethods) {
      if (typeof service[methodName] === 'function') {
        return await service[methodName](promptName, promptArgs);
      }
    }

    // Try dynamic method based on prompt name
    const promptMethodName = this.getPromptMethodName(promptName);
    if (typeof service[promptMethodName] === 'function') {
      return await service[promptMethodName](promptArgs);
    }

    throw new Error(`Service does not support prompt retrieval for: ${promptName}`);
  }

  /**
   * Convert prompt name to method name
   */
  getPromptMethodName(promptName) {
    // Convert prompt_name to promptName or getPromptName
    const camelCase = promptName.replace(/_([a-z])/g, (_, letter) => letter.toUpperCase());
    return `get${camelCase.charAt(0).toUpperCase() + camelCase.slice(1)}Prompt`;
  }

  /**
   * Validate prompt arguments
   */
  validatePromptArguments(promptName, promptArgs, promptSchema) {
    if (!promptSchema || !promptSchema.arguments) {
      return { valid: true };
    }

    const requiredArgs = promptSchema.arguments.filter(arg => arg.required);

    // Check required arguments
    for (const requiredArg of requiredArgs) {
      if (!(requiredArg.name in promptArgs)) {
        return {
          valid: false,
          error: `Missing required argument: ${requiredArg.name}`
        };
      }
    }

    return { valid: true };
  }

  /**
   * Refresh prompt discovery
   */
  async refresh() {
    this.logger.info('Refreshing prompt services');

    this.promptServices.clear();
    await this.discoverPromptServices();

    this.logger.info('Prompt services refreshed');

    return {
      success: true,
      serviceCount: this.promptServices.size,
      promptCount: this.getTotalPromptCount()
    };
  }

  /**
   * Get total number of prompts
   */
  getTotalPromptCount() {
    let count = 0;
    for (const { prompts } of this.promptServices.values()) {
      count += prompts.length;
    }
    return count;
  }

  /**
   * Get prompt statistics
   */
  getStats() {
    const stats = {
      serviceCount: this.promptServices.size,
      promptCount: this.getTotalPromptCount(),
      services: []
    };

    for (const [serviceName, { prompts, metadata }] of this.promptServices) {
      stats.services.push({
        name: serviceName,
        type: metadata.type,
        version: metadata.version,
        promptCount: prompts.length,
        prompts: prompts.map(p => ({
          name: p.name,
          description: p.description,
          argumentCount: p.arguments ? p.arguments.length : 0
        }))
      });
    }

    return stats;
  }

  /**
   * Search prompts by keyword
   */
  searchPrompts(keyword) {
    const results = [];
    const searchTerm = keyword.toLowerCase();

    for (const [serviceName, { prompts }] of this.promptServices) {
      for (const prompt of prompts) {
        const matchesName = prompt.name.toLowerCase().includes(searchTerm);
        const matchesDescription = prompt.description &&
          prompt.description.toLowerCase().includes(searchTerm);

        if (matchesName || matchesDescription) {
          results.push({
            ...prompt,
            _service: serviceName,
            _matchScore: this.calculateMatchScore(prompt, searchTerm)
          });
        }
      }
    }

    // Sort by match score
    return results.sort((a, b) => b._matchScore - a._matchScore);
  }

  /**
   * Calculate match score for search results
   */
  calculateMatchScore(prompt, searchTerm) {
    let score = 0;

    // Exact name match gets highest score
    if (prompt.name.toLowerCase() === searchTerm) {
      score += 100;
    } else if (prompt.name.toLowerCase().includes(searchTerm)) {
      score += 50;
    }

    // Description match gets medium score
    if (prompt.description && prompt.description.toLowerCase().includes(searchTerm)) {
      score += 25;
    }

    return score;
  }

  /**
   * Get prompts by category
   */
  getPromptsByCategory(category) {
    const prompts = [];

    for (const [serviceName, { prompts: servicePrompts }] of this.promptServices) {
      for (const prompt of servicePrompts) {
        if (prompt.category === category) {
          prompts.push({
            ...prompt,
            _service: serviceName
          });
        }
      }
    }

    return prompts;
  }

  /**
   * Health check
   */
  isHealthy() {
    return this.isInitialized && this.serviceRegistry.isHealthy();
  }

  /**
   * Get prompt handler status
   */
  getStatus() {
    return {
      initialized: this.isInitialized,
      serviceCount: this.promptServices.size,
      promptCount: this.getTotalPromptCount(),
      healthy: this.isHealthy(),
      timestamp: new Date().toISOString()
    };
  }

  /**
   * Cleanup
   */
  async cleanup() {
    this.logger.info('Cleaning up Prompt Handler');

    this.promptServices.clear();
    this.isInitialized = false;

    this.logger.info('Prompt Handler cleanup completed');
  }
}

export default PromptHandler;