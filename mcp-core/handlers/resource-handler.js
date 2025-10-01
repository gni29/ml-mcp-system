/**
 * Resource Handler - Manages MCP resource requests
 * Handles resource listing and reading through the service registry
 */

export class ResourceHandler {
  constructor(serviceRegistry, logger) {
    this.serviceRegistry = serviceRegistry;
    this.logger = logger;
    this.resourceServices = new Map();
    this.isInitialized = false;
  }

  /**
   * Initialize the resource handler
   */
  async initialize() {
    try {
      this.logger.info('Initializing Resource Handler');

      // Discover and register resource services
      await this.discoverResourceServices();

      this.isInitialized = true;
      this.logger.info('Resource Handler initialized successfully');

    } catch (error) {
      this.logger.error('Failed to initialize Resource Handler:', error);
      throw error;
    }
  }

  /**
   * Discover resource services from the service registry
   */
  async discoverResourceServices() {
    // Find services that provide resources
    const resourceProviders = this.serviceRegistry.findByCapability('resources');

    this.logger.info(`Discovered ${resourceProviders.length} resource provider services`);

    for (const { name, service, metadata } of resourceProviders) {
      try {
        // Get resources from the service
        const resources = await this.getResourcesFromService(service);

        if (resources && resources.length > 0) {
          this.resourceServices.set(name, {
            service,
            resources,
            metadata
          });

          this.logger.info(`Registered ${resources.length} resources from service: ${name}`);
        }

      } catch (error) {
        this.logger.error(`Failed to get resources from service ${name}:`, error);
      }
    }
  }

  /**
   * Get resources from a service
   */
  async getResourcesFromService(service) {
    if (typeof service.getResources === 'function') {
      return await service.getResources();
    }

    if (typeof service.listResources === 'function') {
      return await service.listResources();
    }

    if (service.resources && Array.isArray(service.resources)) {
      return service.resources;
    }

    return [];
  }

  /**
   * List all available resources
   */
  async listResources(request) {
    if (!this.isInitialized) {
      throw new Error('Resource handler not initialized');
    }

    try {
      const allResources = [];

      // Collect resources from all registered services
      for (const [serviceName, { resources, metadata }] of this.resourceServices) {
        for (const resource of resources) {
          allResources.push({
            ...resource,
            _service: serviceName,
            _serviceType: metadata.type,
            _serviceVersion: metadata.version
          });
        }
      }

      this.logger.info(`Listed ${allResources.length} resources from ${this.resourceServices.size} services`);

      return {
        resources: allResources
      };

    } catch (error) {
      this.logger.error('Failed to list resources:', error);
      throw error;
    }
  }

  /**
   * Read a specific resource
   */
  async readResource(request) {
    if (!this.isInitialized) {
      throw new Error('Resource handler not initialized');
    }

    try {
      const { uri } = request.params;

      if (!uri) {
        throw new Error('Resource URI is required');
      }

      this.logger.info(`Reading resource: ${uri}`);

      // Find the service that provides this resource
      const { service, serviceName } = await this.findResourceService(uri);

      if (!service) {
        throw new Error(`Resource '${uri}' not found`);
      }

      // Read the resource
      const result = await this.readResourceFromService(service, uri);

      this.logger.info(`Resource '${uri}' read successfully`);

      return result;

    } catch (error) {
      this.logger.error(`Failed to read resource '${request.params.uri}':`, error);
      throw error;
    }
  }

  /**
   * Find the service that provides a specific resource
   */
  async findResourceService(uri) {
    for (const [serviceName, { service, resources }] of this.resourceServices) {
      const resource = resources.find(r => r.uri === uri);
      if (resource) {
        return { service, serviceName, resource };
      }
    }

    return { service: null, serviceName: null, resource: null };
  }

  /**
   * Read a resource from a specific service
   */
  async readResourceFromService(service, uri) {
    // Try different method names for resource reading
    const readMethods = [
      'readResource',
      'getResource',
      'fetchResource',
      'loadResource'
    ];

    for (const methodName of readMethods) {
      if (typeof service[methodName] === 'function') {
        return await service[methodName](uri);
      }
    }

    // Try dynamic method based on URI scheme
    const scheme = this.extractUriScheme(uri);
    if (scheme) {
      const schemeMethodName = `read${scheme.charAt(0).toUpperCase() + scheme.slice(1)}Resource`;
      if (typeof service[schemeMethodName] === 'function') {
        return await service[schemeMethodName](uri);
      }
    }

    throw new Error(`Service does not support resource reading for: ${uri}`);
  }

  /**
   * Extract scheme from URI
   */
  extractUriScheme(uri) {
    const match = uri.match(/^([a-zA-Z][a-zA-Z0-9+.-]*):\/\//);
    return match ? match[1] : null;
  }

  /**
   * Refresh resource discovery
   */
  async refresh() {
    this.logger.info('Refreshing resource services');

    this.resourceServices.clear();
    await this.discoverResourceServices();

    this.logger.info('Resource services refreshed');

    return {
      success: true,
      serviceCount: this.resourceServices.size,
      resourceCount: this.getTotalResourceCount()
    };
  }

  /**
   * Get total number of resources
   */
  getTotalResourceCount() {
    let count = 0;
    for (const { resources } of this.resourceServices.values()) {
      count += resources.length;
    }
    return count;
  }

  /**
   * Get resource statistics
   */
  getStats() {
    const stats = {
      serviceCount: this.resourceServices.size,
      resourceCount: this.getTotalResourceCount(),
      services: []
    };

    for (const [serviceName, { resources, metadata }] of this.resourceServices) {
      stats.services.push({
        name: serviceName,
        type: metadata.type,
        version: metadata.version,
        resourceCount: resources.length,
        resources: resources.map(r => ({
          uri: r.uri,
          name: r.name,
          mimeType: r.mimeType
        }))
      });
    }

    return stats;
  }

  /**
   * Validate resource URI
   */
  validateResourceUri(uri) {
    if (!uri || typeof uri !== 'string') {
      return {
        valid: false,
        error: 'URI must be a non-empty string'
      };
    }

    // Basic URI validation - can be extended
    if (!uri.includes('://')) {
      return {
        valid: false,
        error: 'URI must include scheme (e.g., protocol://)'
      };
    }

    return { valid: true };
  }

  /**
   * Get resources by type
   */
  getResourcesByType(mimeType) {
    const resources = [];

    for (const [serviceName, { resources: serviceResources }] of this.resourceServices) {
      for (const resource of serviceResources) {
        if (resource.mimeType === mimeType) {
          resources.push({
            ...resource,
            _service: serviceName
          });
        }
      }
    }

    return resources;
  }

  /**
   * Get resources by scheme
   */
  getResourcesByScheme(scheme) {
    const resources = [];

    for (const [serviceName, { resources: serviceResources }] of this.resourceServices) {
      for (const resource of serviceResources) {
        const resourceScheme = this.extractUriScheme(resource.uri);
        if (resourceScheme === scheme) {
          resources.push({
            ...resource,
            _service: serviceName
          });
        }
      }
    }

    return resources;
  }

  /**
   * Health check
   */
  isHealthy() {
    return this.isInitialized && this.serviceRegistry.isHealthy();
  }

  /**
   * Get resource handler status
   */
  getStatus() {
    return {
      initialized: this.isInitialized,
      serviceCount: this.resourceServices.size,
      resourceCount: this.getTotalResourceCount(),
      healthy: this.isHealthy(),
      timestamp: new Date().toISOString()
    };
  }

  /**
   * Cleanup
   */
  async cleanup() {
    this.logger.info('Cleaning up Resource Handler');

    this.resourceServices.clear();
    this.isInitialized = false;

    this.logger.info('Resource Handler cleanup completed');
  }
}

export default ResourceHandler;