/**
 * Validation Middleware - Request validation for MCP handlers
 * Validates incoming requests before they reach the handlers
 */

export class ValidationMiddleware {
  constructor() {
    this.validationRules = new Map();
    this.setupDefaultRules();
  }

  /**
   * Setup default validation rules for MCP requests
   */
  setupDefaultRules() {
    // Tools validation rules
    this.validationRules.set('tools/list', {
      requiresParams: false,
      allowedParams: []
    });

    this.validationRules.set('tools/call', {
      requiresParams: true,
      requiredParams: ['name'],
      allowedParams: ['name', 'arguments'],
      paramTypes: {
        name: 'string',
        arguments: 'object'
      }
    });

    // Resources validation rules
    this.validationRules.set('resources/list', {
      requiresParams: false,
      allowedParams: []
    });

    this.validationRules.set('resources/read', {
      requiresParams: true,
      requiredParams: ['uri'],
      allowedParams: ['uri'],
      paramTypes: {
        uri: 'string'
      }
    });

    // Prompts validation rules
    this.validationRules.set('prompts/list', {
      requiresParams: false,
      allowedParams: []
    });

    this.validationRules.set('prompts/get', {
      requiresParams: true,
      requiredParams: ['name'],
      allowedParams: ['name', 'arguments'],
      paramTypes: {
        name: 'string',
        arguments: 'object'
      }
    });
  }

  /**
   * Validate an MCP request
   */
  async validateRequest(handlerName, request) {
    const rules = this.validationRules.get(handlerName);

    if (!rules) {
      // No specific validation rules, allow request
      return { valid: true };
    }

    try {
      // Validate request structure
      this.validateRequestStructure(request);

      // Validate parameters
      this.validateParameters(request.params || {}, rules);

      return { valid: true };

    } catch (error) {
      throw new ValidationError(`Validation failed for ${handlerName}: ${error.message}`);
    }
  }

  /**
   * Validate basic request structure
   */
  validateRequestStructure(request) {
    if (!request) {
      throw new Error('Request cannot be null or undefined');
    }

    if (typeof request !== 'object') {
      throw new Error('Request must be an object');
    }

    // MCP requests should have certain properties
    if (!request.method && !request.params) {
      throw new Error('Request must have method or params property');
    }
  }

  /**
   * Validate request parameters
   */
  validateParameters(params, rules) {
    // Check if parameters are required
    if (rules.requiresParams && (!params || Object.keys(params).length === 0)) {
      throw new Error('Parameters are required for this request');
    }

    // Check required parameters
    if (rules.requiredParams) {
      for (const requiredParam of rules.requiredParams) {
        if (!(requiredParam in params)) {
          throw new Error(`Missing required parameter: ${requiredParam}`);
        }
      }
    }

    // Check parameter types
    if (rules.paramTypes) {
      for (const [paramName, expectedType] of Object.entries(rules.paramTypes)) {
        if (paramName in params) {
          const actualType = this.getParameterType(params[paramName]);
          if (actualType !== expectedType) {
            throw new Error(`Parameter '${paramName}' must be of type ${expectedType}, got ${actualType}`);
          }
        }
      }
    }

    // Check allowed parameters
    if (rules.allowedParams) {
      for (const paramName of Object.keys(params)) {
        if (!rules.allowedParams.includes(paramName)) {
          throw new Error(`Unknown parameter: ${paramName}`);
        }
      }
    }

    // Custom validations
    if (rules.customValidations) {
      for (const validation of rules.customValidations) {
        validation(params);
      }
    }
  }

  /**
   * Get parameter type for validation
   */
  getParameterType(value) {
    if (value === null) {
      return 'null';
    }

    if (Array.isArray(value)) {
      return 'array';
    }

    return typeof value;
  }

  /**
   * Add custom validation rule
   */
  addValidationRule(handlerName, rules) {
    this.validationRules.set(handlerName, {
      requiresParams: false,
      allowedParams: [],
      ...rules
    });
  }

  /**
   * Add custom validation function
   */
  addCustomValidation(handlerName, validationFunction) {
    const existingRules = this.validationRules.get(handlerName) || {};

    if (!existingRules.customValidations) {
      existingRules.customValidations = [];
    }

    existingRules.customValidations.push(validationFunction);
    this.validationRules.set(handlerName, existingRules);
  }

  /**
   * Validate tool-specific parameters
   */
  validateToolParameters(toolName, toolArgs, toolSchema) {
    if (!toolSchema || !toolSchema.inputSchema) {
      return { valid: true };
    }

    const schema = toolSchema.inputSchema;

    try {
      // Validate required properties
      if (schema.required) {
        for (const requiredField of schema.required) {
          if (!(requiredField in toolArgs)) {
            throw new Error(`Missing required parameter: ${requiredField}`);
          }
        }
      }

      // Validate property types
      if (schema.properties) {
        for (const [propName, propSchema] of Object.entries(schema.properties)) {
          if (propName in toolArgs) {
            this.validatePropertyType(propName, toolArgs[propName], propSchema);
          }
        }
      }

      return { valid: true };

    } catch (error) {
      throw new ValidationError(`Tool parameter validation failed: ${error.message}`);
    }
  }

  /**
   * Validate individual property type
   */
  validatePropertyType(propName, value, propSchema) {
    const { type, enum: enumValues, minimum, maximum } = propSchema;

    // Type validation
    if (type) {
      const actualType = this.getParameterType(value);

      if (type === 'integer' && actualType === 'number' && Number.isInteger(value)) {
        // Integer is valid
      } else if (actualType !== type) {
        throw new Error(`Property '${propName}' must be of type ${type}, got ${actualType}`);
      }
    }

    // Enum validation
    if (enumValues && !enumValues.includes(value)) {
      throw new Error(`Property '${propName}' must be one of: ${enumValues.join(', ')}`);
    }

    // Number range validation
    if (type === 'number' || type === 'integer') {
      if (minimum !== undefined && value < minimum) {
        throw new Error(`Property '${propName}' must be >= ${minimum}`);
      }
      if (maximum !== undefined && value > maximum) {
        throw new Error(`Property '${propName}' must be <= ${maximum}`);
      }
    }
  }

  /**
   * Validate resource URI format
   */
  validateResourceUri(uri) {
    if (!uri || typeof uri !== 'string') {
      throw new Error('URI must be a non-empty string');
    }

    // Basic URI format validation
    if (!uri.includes('://')) {
      throw new Error('URI must include a scheme (e.g., protocol://)');
    }

    // Additional URI validations can be added here
    const uriPattern = /^[a-zA-Z][a-zA-Z0-9+.-]*:\/\/.+/;
    if (!uriPattern.test(uri)) {
      throw new Error('URI format is invalid');
    }

    return { valid: true };
  }

  /**
   * Validate prompt arguments
   */
  validatePromptArguments(promptName, promptArgs, promptSchema) {
    if (!promptSchema || !promptSchema.arguments) {
      return { valid: true };
    }

    try {
      const requiredArgs = promptSchema.arguments.filter(arg => arg.required);

      // Check required arguments
      for (const requiredArg of requiredArgs) {
        if (!(requiredArg.name in promptArgs)) {
          throw new Error(`Missing required argument: ${requiredArg.name}`);
        }
      }

      // Validate argument types (if specified)
      for (const argSchema of promptSchema.arguments) {
        if (argSchema.name in promptArgs && argSchema.type) {
          const actualType = this.getParameterType(promptArgs[argSchema.name]);
          if (actualType !== argSchema.type) {
            throw new Error(`Argument '${argSchema.name}' must be of type ${argSchema.type}, got ${actualType}`);
          }
        }
      }

      return { valid: true };

    } catch (error) {
      throw new ValidationError(`Prompt argument validation failed: ${error.message}`);
    }
  }

  /**
   * Sanitize input to prevent injection attacks
   */
  sanitizeInput(input) {
    if (typeof input === 'string') {
      // Basic sanitization - remove potential harmful characters
      return input.replace(/[<>\"'&]/g, '');
    }

    if (typeof input === 'object' && input !== null) {
      const sanitized = {};
      for (const [key, value] of Object.entries(input)) {
        sanitized[this.sanitizeInput(key)] = this.sanitizeInput(value);
      }
      return sanitized;
    }

    return input;
  }

  /**
   * Get validation statistics
   */
  getValidationStats() {
    return {
      rulesCount: this.validationRules.size,
      handlers: Array.from(this.validationRules.keys()),
      timestamp: new Date().toISOString()
    };
  }

  /**
   * Get validation rules for debugging
   */
  getValidationRules() {
    return Object.fromEntries(this.validationRules);
  }
}

/**
 * Custom validation error class
 */
export class ValidationError extends Error {
  constructor(message) {
    super(message);
    this.name = 'ValidationError';
  }
}

export default ValidationMiddleware;